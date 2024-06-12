import tsgen

p1=tsgen.TSGen("param_data.json")

hsh = input('What is the hash of SQL whose Timew Series you want to extract ? ')

nbr = p1.mainProcess(hsh)

