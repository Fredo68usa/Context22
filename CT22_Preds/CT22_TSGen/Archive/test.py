#    for hit in A_SEL_TYP_tmp['hits']['hits']:
           # print( hit["_source"])
myListSelectType = []
myListSelectType.append( ["_source","Hello", "Bonjour"])
myListSelectType.append( ["_source","Hello", "Bonjour"])
print (myListSelectType)
import pandas as pd

# products_list = [['laptop',1300],['printer',150],['tablet',300],['desk',450],['chair',200]]

df = pd.DataFrame (myListSelectType, columns = ['source', 'anglais', 'francais'])
print (df)
