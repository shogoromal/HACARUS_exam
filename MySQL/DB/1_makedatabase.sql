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
age FLOAT,
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

CREATE TABLE setting (
setting_id INT AUTO_INCREMENT PRIMARY KEY,
best_model_id INT,
num_for_result_check INT,
check_type TEXT,
threshold_percentage FLOAT,
num_for_re_training INT
);

INSERT INTO 
setting (best_model_id, num_for_result_check, check_type, threshold_percentage, num_for_re_training)
VALUE (1, 10, "accuracy", 0.5, 40);

CREATE TABLE health_check_log (
health_check_id INT AUTO_INCREMENT PRIMARY KEY,
model_version_id INT,
data_id INT,
setting_id INT,
FOREIGN KEY (model_version_id) REFERENCES models (model_version_id),
FOREIGN KEY (data_id) REFERENCES datas (data_id),
FOREIGN KEY (setting_id) REFERENCES setting (setting_id),
accuracy_score FLOAT,
f1_score FLOAT,
NG_decision BOOLEAN
);