import ekfuzzy
import threading
p1 = ekfuzzy.EKFuzzy()

if p1.tbcSQL is None:
   p1.tbcSQL = "Select Hello fom CCN "


print ('--------------------')
print (p1.tbcSQL)
print ('--------------------')


thread = threading.Thread(target=p1.main_process())
thread.start()

