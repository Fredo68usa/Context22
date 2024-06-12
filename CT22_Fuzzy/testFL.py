print ('Tested against ' , row[1] , "Standard Fuzzy score :" ,fuzz.ratio(self.tbcSQL, refSQL), "-- Partial Fuzzy score :" ,fuzz.partial_ratio(self.tbcSQL, refSQL))

python3 main.py "SELECT * FROM HIPAAHOSPITAL"
 1028  python3 main.py "Select * from HIPAA where ssn = '?'"
 1030  python3 main.py "Select * from HIPAA where ssn = '??'"
 1032  python3 main.py "SELECT * FROM CCNTBL"
 1038  python3 main.py "SELECT * FROM BANKACCOUNTBB"
 1048  python3 main.py "SELECT * FROM BANKACCOUNTCCC"
 1056  python3 main.py "SELECT * FROM BANKACCOUNTXXXXX"
 1064  python3 main.py "SELECT * FROM BANKXXXXX"
 1099  python3 main.py "SELECT * FROM HIPAAHOSPITAL"

