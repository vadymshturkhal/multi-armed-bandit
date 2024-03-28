DB_NAME="multi_armed"
USER="agent"
PASSWORD="5271"
HOST="localhost"
PORT=5432

"""
CREATE TABLE epochs (
    epoch_id SERIAL PRIMARY KEY,
    description TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

"""
CREATE TABLE average_data (
    data_id SERIAL PRIMARY KEY,
    step INT,
    bet FLOAT,
    reward FLOAT,
    points FLOAT,
    average FLOAT,
    fk_epoch_id INT,
    FOREIGN KEY (fk_epoch_id) REFERENCES epochs(epoch_id)
);
"""

"GRANT ALL PRIVILEGES ON DATABASE multi_armed TO agent;"
"GRANT ALL PRIVILEGES ON TABLE average_data TO agent;"
"GRANT ALL PRIVILEGES ON TABLE epochs TO agent;"
"GRANT USAGE, SELECT ON SEQUENCE average_data_data_id_seq TO agent;"
"GRANT USAGE, SELECT ON SEQUENCE epochs_epoch_id_seq TO agent;"

"""
ALTER TABLE average_data ADD CONSTRAINT fk_epoch_id FOREIGN KEY (epoch_id) REFERENCES epochs(epoch_id) ON DELETE CASCADE;
"""