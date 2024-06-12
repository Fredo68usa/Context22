import ekfuzzy
import threading
p1 = ekfuzzy.EKFuzzy()
# p1.get_EKFuzzy_details()
# p1.posGresPrep()

if p1.tbcSQL is None:
   p1.tbcSQL = "Select Hello fom CCN "


print ('--------------------')
print (p1.tbcSQL)
print ('--------------------')


# p1.checkFuzzy()
# thread = threading.Thread(target=p1.checkFuzzy())
thread = threading.Thread(target=p1.main_process())
thread.start()

# p1.close_PosGres()
