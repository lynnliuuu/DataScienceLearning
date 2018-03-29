import pprint
import re
import xml.etree.cElementTree as ET
from collections import defaultdict
import codecs
import sys
stdout = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = stdout

'''
u = u'1006\u5f04'
s = u.encode('utf-8')
print s
英文 re.compile(r'\b\S+\.?$', re.IGNORECASE)
'''

OSM_FILE = "shanghai_major.osm"


'''
"Xingping Road"
addr:city" v="Shanghai"

纯中文的错：'弄'："军工路 516 弄" "金坛路35弄" "平兴线庄兴路段"
纯英文的错："Zhenxun Rd" "Tomson Golf Garden"
中英一起的错： "addr:street" "玉古路 Yugu Road" "延吉东路82弄 Lane 82 of East Yanji Road" "松花一村 Songhua Community #1"
市、区错:
k="addr:city" v="苏州" /> <tag k="addr:street" v="ren'ai road"/>
k="addr:city" v="Hudai Town" />
k="addr:city" v="松江区" />
k="addr:city" v="上海市" />
		<tag k="addr:district" v="嘉定区" />
		<tag k="addr:housenumber" v="4800" />
		<tag k="addr:street" v="曹安公路" />
k="addr:city" v="新埭镇" />
'''
pattern1 = ur'[\u4E00-\u9FA5]+'
pattern2 = ur'\b\S+\.?$'

street_type_re1 = re.compile(pattern1)
street_type_re2 = re.compile(pattern2, re.IGNORECASE)


street_types = defaultdict(set)
expected = [u"段", u"路", u"道", u"线", u"街"]


def audit_street_type(street_types, street_name):
	if street_type_re1.search(street_name):
		street_type = street_type_re1.search(street_name).group()
		if street_type[-1] not in expected:
			street_types[street_type[-1]].add(street_name)
	elif street_type_re2.search(street_name):
		street_type = street_type_re2.search(street_name).group()
		street_types[street_type].add(street_name)


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def print_sorted_dict(d):
    keys = d.keys()
    keys = sorted(keys, key=lambda s: s.lower())
    for k in keys:
        v = d[k]
        print "%s: %d" % {k, v}

#访问道路的标签
#def audit():
for event,elem in ET.iterparse(OSM_FILE, events=("start",)):
    if elem.tag == "way":
        for tag in elem.iter("tag"):
            if is_street_name(tag):
                audit_street_type(street_types, tag.attrib['v'])

dic = dict(street_types)
for key,val in dic.items():
	val = list(val)
	print str(key + ": " + val[0])

#print dict(street_types)

#print_sorted_dict(street_types)
