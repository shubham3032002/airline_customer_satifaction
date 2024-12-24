import os
import csv
import mysql.connector
from mysql.connector import Error
from config.db_config import DB_CONFIG
from src.logger import logger
from src.exception import MyException
import sys
import pymysql

class MySQLDataExtractor:
    def __init__(self, db_config):
        self.host = db_config["host"]
        self.user = db_config["user"]
        self.password = db_config["password"]
        self.database = db_config["database"]
        self.table_name = db_config["table_name"]
        self.connection = None

        logger.info("Your Database configuration has been set up")

    def connect(self):
        try:
            self.connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
            )
            if self.connection.open:
                logger.info("Successfully connected to the Database")
        except pymysql.MySQLError as e:
            raise MyException(f"Error while connecting to the Database: {e}", sys)

    def disconnect(self):
            if self.connection:
                self.connection.close()
                logger.info("Disconnected from the Database")


    def extract_to_csv(self, output_folder="./artifacts/raw"):
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()

            cursor = self.connection.cursor()
            query = f"SELECT * FROM {self.table_name}"
            cursor.execute(query)

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            logger.info("Data fetched successfully!")

            os.makedirs(output_folder, exist_ok=True)
            csv_file_path = os.path.join(output_folder, "data.csv")

            with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(columns)
                writer.writerows(rows)

                logger.info(f"Data successfully saved to {csv_file_path}")

        except Error as e:
            raise MyException(f"Error in extracting DB due to SQL: {e}", sys)

        except MyException as ce:
            logger.error(str(ce))
            raise  # Re-raise to propagate the exception if necessary

        finally:
            if "cursor" in locals():
                cursor.close()
            self.disconnect()


if __name__ == "__main__":
    try:
        extractor = MySQLDataExtractor(DB_CONFIG)
        extractor.extract_to_csv()
    except MyException as ce:
        logger.error(str(ce))
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
