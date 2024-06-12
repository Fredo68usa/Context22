DELETE from currentsqls where hash = '1ac96c2294ebf794b35ab4a32eb17142';
DELETE from currentsqls where hash = '05b7d25521aafc231b5876d8b5a7ac27';
DELETE from currentsqls where hash = '8bee66e54fa8dc195683fa9185647c1a';
DELETE from currentsqls where hash = 'ef0d91583437a407e7208f5cd079da02';

INSERT INTO currentsqls VALUES ( md5('SELECT * FROM HIPAA22 where'), 'SELECT * FROM HIPAA22 where', 'Pending' , 1300);
INSERT INTO currentsqls VALUES ( md5('SELECT * FROM HIPAA23 where'), 'SELECT * FROM HIPAA23 where', 'Pending' , 1200);
INSERT INTO currentsqls VALUES ( md5('SELECT * FROM HIPAA24 where'), 'SELECT * FROM HIPAA24 where', 'Pending' , 1100);
INSERT INTO currentsqls VALUES ( md5('SELECT * FROM HIPAA25 where'), 'SELECT * FROM HIPAA25 where', 'Pending' , 1000);
