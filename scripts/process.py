import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "gender-guesser", "gensim", "xlrd"])

import argparse
import logging
import os
import re
import string

import gender_guesser.detector as gender
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def parse_args():
    """Parse job arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args, _ = parser.parse_known_args()
    logger.info(f"Received arguments {args}")
    return args


def load_data():
    """Load data from input path"""
    input_path = os.path.join("/opt/ml/processing/hospital_data", "basededatos_50K.xlsx")
    logger.info(f"Loading data from {input_path}")
    df = pd.read_excel(input_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    logger.info(f"Dataset info: {df.info()}")
    return df


def clean_duplicates(df):
    """Remove duplicate records"""
    logger.info("Cleaning duplicates")
    initial_len = len(df)
    df = df.drop_duplicates()
    logger.info(
        f"Number of duplicates:\t\t {initial_len - len(df)} - {(initial_len - len(df)) / len(df) * 100:.1f}%"
    )
    return df


def handle_missing_values(df):
    """Handle missing values in the dataset"""
    logger.info("Handling missing values")
    initial_len = len(df)
    df = df.dropna()
    logger.info(
        f"Number of missing values:\t {initial_len - len(df)} - {(initial_len - len(df)) / len(df) * 100:.1f}%"
    )
    return df


def absolute_values(df):
    """Convert values to absolute values"""
    logger.info("Converting values to absolute values")
    df["tiempo_espera_triage"] = df["tiempo_espera_triage"].abs()
    df["tiempo_en_triage"] = df["tiempo_en_triage"].abs()
    df["tiempo_espera_despuestriage"] = df["tiempo_espera_despuestriage"].abs()
    df["tiempo_en_consulta"] = df["tiempo_en_consulta"].abs()
    df["edad"] = df["edad"].abs()

    # create a column with the total time
    df["tiempo_total"] = (
        df["tiempo_espera_triage"]
        + df["tiempo_en_triage"]
        + df["tiempo_espera_despuestriage"]
        + df["tiempo_en_consulta"]
    )
    return df


def remove_outliers(df, column):
    """Remove outliers from a column"""
    logger.info(f"Removing outliers from {column}")
    initial_len = len(df)

    if column == "edad":
        logger.info("Standard age values")
        df.rename(columns={"Edad1": "edad_unidad"}, inplace=True)
        # convert column edad_unidad to numeric.
        df["edad_unidad"] = df["edad_unidad"].replace(
            {"Años": 1, "Meses": 1 / 12, "Dias": 1 / 365}
        )
        # standardize the value of the age
        df["edad"] = df["edad"] * df["edad_unidad"]

    # Calculate the percentiles
    twenty_fifth = df[column].quantile(0.25)
    seventy_fifth = df[column].quantile(0.75)

    # Obtain IQR
    iqr = seventy_fifth - twenty_fifth

    # Upper and lower thresholds
    upper = seventy_fifth + (1.5 * iqr)
    lower = twenty_fifth - (1.5 * iqr)

    # Subset the dataset
    outliers = df[(df[column] < lower) | (df[column] > upper)]

    logger.info(
        f"Number of outliers for {column}:\t {len(outliers)} - {(len(outliers) / initial_len * 100):.1f}%"
    )

    df = df.drop(outliers.index)

    return df


def gender_encoding(df):
    """Encode gender"""
    logger.info("Encoding gender")
    initial_len = len(df)
    # Create an gender detector object
    d = gender.Detector()

    def obtener_genero(nombre):
        # setting the second name, there is a total categorization between 'Ambiguo' y 'Desconocido' of 34,535 names
        nombre = nombre.split()[-1]
        nombre = nombre.title()
        # if the name is 'Femenino' or 'Masculino' return the name
        # total categorization between 'Ambiguo' y 'Desconocido' of 27,709 names. Recover 6,826 names
        if nombre == "Femenino" or nombre == "Masculino":
            return nombre
        else:
            return d.get_gender(nombre)

    df["genero"] = df["nombre"].apply(obtener_genero)

    df["genero"] = df["genero"].replace(
        {
            "male": "Masculino",
            "female": "Femenino",
            "andy": "Ambiguo",
            "unknown": "Desconocido",
            "mostly_male": "Masculino",
            "mostly_female": "Femenino",
        }
    )

    df_ambiguo = df[(df["genero"] == "Ambiguo") | (df["genero"] == "Desconocido")]

    logger.info(
        f"Number of unknown:\t {len(df_ambiguo)} - {len(df_ambiguo) / initial_len * 100:.1f}%"
    )

    df = df.drop(df_ambiguo.index)
    return df


def encode_categorical(df):
    """Encode categorical variables"""
    logger.info("Encoding categorical variables")
    # Encode clasification 'ROJO': 0, 'AMARILLO': 1, 'VERDE': 2
    df["clasificacion_encode"] = df["clasificacion"].replace(
        {"ROJO": 0, "AMARILLO": 1, "VERDE": 2}
    )

    encoder = OneHotEncoder(sparse_output=False)
    # Aplicar el codificador a la columna 'genero'
    genero_encoded = encoder.fit_transform(df[["genero"]])
    # Convertir la salida a un DataFrame
    genero_encoded_df = pd.DataFrame(
        genero_encoded, columns=encoder.get_feature_names_out(["genero"]), index=df.index
    )
    # Concatenar las columnas codificadas con el DataFrame original
    df = pd.concat([df, genero_encoded_df], axis=1)

    return df


def process_dates(df):
    """Extract features from date columns"""
    logger.info("Extracting time features from date columns")
    # Extract caracteristics from date
    df["hora"] = df["Fecha"].dt.hour
    df["minuto"] = df["Fecha"].dt.minute
    df["mes"] = df["Fecha"].dt.month
    df["dia"] = df["Fecha"].dt.day
    df["dia_semana"] = df["Fecha"].dt.dayofweek

    return df


def remove_punctuation(df):
    """Remove punctuation from the dataset"""
    logger.info("Removing punctuation from the dataset")
    initial_len = len(df)
    # clean text
    df_not_string = df[~df["Dx"].apply(lambda x: isinstance(x, str))]
    logger.info(
        f"Number of not string values:\t {len(df_not_string)} - {len(df_not_string) / initial_len * 100:.3f}%"
    )
    df = df.drop(df_not_string.index)

    def remove_punctuation(text):
        """custom function to remove the punctuation"""
        return text.translate(str.maketrans("", "", string.punctuation + "1" + "2" + "3"))

    df["Dx"] = df["Dx"].apply(lambda text: remove_punctuation(text))

    return df


def remove_empty_string(df):
    """Remove empty string from the dataset"""
    logger.info("Removing empty string from the dataset")
    initial_len = len(df)
    df_empty_string = df[df["Dx"] == ""]
    logger.info(
        f"Number of empty string:\t {len(df_empty_string)} - {len(df_empty_string) / initial_len * 100:.1f}%"
    )
    df = df.drop(df_empty_string.index)
    return df


def delete_meaningless_strings(df):
    """Delete meaningless strings from the dataset"""
    logger.info("Deleting meaningless strings from the dataset")
    initial_len = len(df)

    df["Dx"] = df["Dx"].str.lower()

    def search_meaningless_str(text):
        found = re.search("^x+$", text)
        if found is not None:
            return True
        else:
            return False

    df_x_strings = df[df["Dx"].apply(search_meaningless_str)]
    logger.info(
        f"Number of meaningless string:\t {len(df_x_strings)} - {len(df_x_strings) / initial_len * 100:.1f}%"
    )

    df = df.drop(df_x_strings.index)

    return df


def generate_embeddings(df):
    """Generate embeddings for the dataset"""
    # Create a TaggedDocument object
    tagged_data = [
        TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(df["Dx"])
    ]

    # Create a Doc2Vec model
    model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=40)

    df["doc2vec"] = [model.dv[str(i)] for i in range(len(tagged_data))]

    df_doc2vec = pd.DataFrame(df["doc2vec"].tolist(), index=df.index)
    df = pd.concat([df.drop("doc2vec", axis=1), df_doc2vec], axis=1)

    return df


if __name__ == "__main__":
    """Main preprocessing function"""
    args = parse_args()

    # Load data
    df = load_data()

    # Clean duplicates
    df = clean_duplicates(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Convert values to absolute values
    df = absolute_values(df)

    # Outliers for 'tiempo_total' (IQR) interquartile range
    df = remove_outliers(df, "tiempo_total")

    # Outliers for 'edad' (IQR) interquartile range
    df = remove_outliers(df, "edad")

    # Obtain gender
    df = gender_encoding(df)

    # Encode categorical variables
    df = encode_categorical(df)

    # Extract time features
    df = process_dates(df)

    # Removal of punctuation
    df = remove_punctuation(df)

    # Remove empty string
    df = remove_empty_string(df)

    # Delete meaningless strings
    df = delete_meaningless_strings(df)

    # Generate embeddings
    df = generate_embeddings(df)

    df = df.drop(
        columns=["nombre", "apat", "amat", "Fecha", "Dx", "edad_unidad", "genero", "clasificacion"]
    )
    logger.info(f"Dropped columns: {df.columns}")
    logger.info(f"Processed data shape: {df.shape}")

    # Split into train, validation, and test sets
    logger.debug("Splitting data into train, validation, and test sets")
    x = df.drop(columns=["tiempo_total"])
    y = df["tiempo_total"]

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=args.test_ratio, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_val, y_val, test_size=args.validation_ratio, random_state=42
    )

    train_df = pd.concat([y_train, x_train], axis=1)
    val_df = pd.concat([y_val, x_val], axis=1)
    test_df = pd.concat([y_test, x_test], axis=1)
    dataset_df = pd.concat([y, x], axis=1)

    logger.info("Train data shape after preprocessing: {}".format(train_df.shape))
    logger.info("Validation data shape after preprocessing: {}".format(val_df.shape))
    logger.info("Test data shape after preprocessing: {}".format(test_df.shape))

    # Save processed datasets to the local paths in the processing container.
    # SageMaker will upload the contents of these paths to S3 bucket
    local_dir = "/opt/ml/processing"
    logger.debug("Writing processed datasets to container local path.")
    train_output_path = os.path.join(f"{local_dir}/train", "train.csv")
    validation_output_path = os.path.join(f"{local_dir}/val", "validation.csv")
    test_output_path = os.path.join(f"{local_dir}/test", "test.csv")
    full_processed_output_path = os.path.join(f"{local_dir}/full", "dataset.csv")

    logger.info("Saving train data to {}".format(train_output_path))
    train_df.to_csv(train_output_path, index=False)

    logger.info("Saving validation data to {}".format(validation_output_path))
    val_df.to_csv(validation_output_path, index=False)

    logger.info("Saving test data to {}".format(test_output_path))
    test_df.to_csv(test_output_path, index=False)

    logger.info("Saving full processed data to {}".format(full_processed_output_path))
    dataset_df.to_csv(full_processed_output_path, index=False)
