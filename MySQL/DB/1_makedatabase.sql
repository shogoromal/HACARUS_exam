CREATE DATABASE mydatabase;
USE mydatabase;

CREATE TABLE models (
model_version_id INT AUTO_INCREMENT PRIMARY KEY,
training_date DATE NOT NULL,
my_model_name TEXT NOT NULL,
pclass_coef FLOAT NOT NULL,
sex_coef FLOAT NOT NULL,
age_coef FLOAT NOT NULL,
fare_coef FLOAT NOT NULL,
training_iteration INT NOT NULL
);

CREATE TABLE datas (
data_id INT AUTO_INCREMENT PRIMARY KEY,
upload_date DATE NOT NULL,
survived INT,
pclass INT,
sex INT,
age INT,
fare FLOAT
);

CREATE TABLE data_model_relations (
experiment_id INT AUTO_INCREMENT PRIMARY KEY,
model_version_id INT,
data_id INT,
FOREIGN KEY (model_version_id) REFERENCES models (model_version_id),
FOREIGN KEY (data_id) REFERENCES datas (data_id),
prediction_score FLOAT,
include_training BOOLEAN
);