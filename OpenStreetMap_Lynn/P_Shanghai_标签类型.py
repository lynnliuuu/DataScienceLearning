import xml.etree.cElementTree as ET
import pprint
import re

lower = re.compile(r'^([a-z]|_)*$') #开始，小写字母或下划线（0或多次），结束
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$') #开始， 小写字母或下划线（0或多次），冒号，小写字母或下划线（0或多次），结束
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
#问题字符，= + / & < > ; ' " ? % # $ @ , .
#\n换行符  \b退格符  \f换页符  \r回车符  \t制表符 \v垂直制表符  \a警报
#\u使用数字指定的Unicode 字符，如\u2000
#\x使用十六进制数指定的Unicode 字符,如\xc8

def key_type(element, keys):
    if element.tag == "tag":
        if re.search(lower,element.get("k")):
            keys["lower"] += 1
        elif re.search(lower_colon,element.get("k")):
            keys["lower_colon"] += 1
        elif re.search(problemchars,element.get("k")):
            keys["problemchars"] += 1
        else:
            keys["other"] += 1
    return keys

def process_map(filename):
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)
    return keys


keys3 = process_map('shanghai_major.osm')
pprint.pprint(keys3)

keys2 = process_map('sample_20.1.osm')
pprint.pprint(keys2)

keys1 = process_map('sample_10.1.osm')
pprint.pprint(keys1)
