INSERT INTO currentsqls VALUES ( md5('SELECT * FROM CCNTBL'), 'SELECT * FROM CCNTBL', 'Pending' ,300);
INSERT INTO currentsqls VALUES ( md5('SELECT * FROM HIPAA'), 'SELECT * FROM HIPAA', 'Pending' ,  400);
INSERT INTO currentsqls VALUES ( md5('SELECT * FROM CCNTBL where CCN = "?"'), 'SELECT * FROM CCNTBL where CCN = "?"', 'Pending',300);
INSERT INTO currentsqls VALUES ( md5('SELECT name, description FROM products WHERE category = "Gifts"'), 'SELECT name, description FROM products WHERE category = "Gifts"', 'Pending',300);
INSERT INTO currentsqls VALUES ( md5('SELECT name, description FROM products WHERE category = "Gifts" UNION SELECT username, password FROM users--'), 'SELECT name, description FROM products WHERE category = "Gifts" UNION SELECT username, password FROM users--', 'Pending',300);
