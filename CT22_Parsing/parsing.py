from icecream import ic


sql = "Select * from TABLE where id = '12345' and id2 = 12345;"

ic(sql)

index = sql.find('=')

ic(index)

index = sql.find('=')

ic(index)

sql.split
