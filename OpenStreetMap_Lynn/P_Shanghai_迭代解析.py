#迭代解析
import xml.etree.cElementTree as ET
from collections import Counter
import pprint


def count_tags(filename):

    osm_file = ET.iterparse(filename, events=('start','end'))
    count = Counter()
    for event, elem in osm_file:
        try:
            if event=='end':
                count[elem.tag] += 1
        except:
            pass
    return count

tags3 = count_tags('sample_20.1.osm')
pprint.pprint(tags3)

tags2 = count_tags('sample_10.1.osm')
pprint.pprint(tags2)

tags1 = count_tags('shanghai_major.osm')
pprint.pprint(tags1)
