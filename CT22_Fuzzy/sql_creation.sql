DROP TABLE similarSQLs;
DROP TABLE currentSQLs;

CREATE TABLE currentSQLs(
hash varchar(32) PRIMARY KEY,
SQL varchar(500),
status varchar(20),
frequency INT
);

CREATE TABLE similarSQLs(
hash varchar(32),
hashSim varchar(32),
status varchar(20),
score INT,
PRIMARY KEY(hash, hashSim),
CONSTRAINT fk_hash
  FOREIGN KEY(hash)
  REFERENCES currentSQLs(hash)
  ON DELETE CASCADE
  ON UPDATE CASCADE
);
