import pprint
import re
import xml.etree.cElementTree as ET
#from collections import defaultdict
import codecs
import csv
import cerberus
import schema
import sys
stdout = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = stdout

OSM_FILE = "shanghai_major2.osm"

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(ur'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(ur'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

SCHEMA = schema.schema

pattern1 = ur'^[a-z\.\(\)\-\,\?]+$'
pattern2 = ur'[\u4E00-\u9FA5]+'
pattern3 = ur'^[\u4E00-\u9FA5]+$'
#pattern4 = ur'^[\u4E00-\u9FA5]+\d+\-*\d*[\u4E00-\u9FA50-9]+$'
#pattern5 = ur'\s'

st_type_re1 = re.compile(pattern1, re.IGNORECASE)
st_type_re2 = re.compile(pattern2, re.IGNORECASE)
st_type_re3 = re.compile(pattern3)
#st_type_re4 = re.compile(pattern4)
#st_type_re5 = re.compile(pattern5)

expected = [u"段", u"路", u"道", u"线", u"街"]
#st_types = defaultdict(set)


#从xml文件中读取所需的标签行，返回一个生成器
def get_element(osm_file, tags=('node', 'way', 'relation')):
    with open(osm_file, "r") as osm:
        context = ET.iterparse(osm, events=('start', 'end'))
        _, root = next(context)
        for event, elem in context:
            if event == 'end' and elem.tag in tags:
                yield elem
                root.clear()


#判断tag是否是 地址：街道
def is_st_name(tag):
    return (tag.attrib['k'] == "addr:street")



def mapping(t,new_name):
    if new_name.find(u"区") != -1:
        #如果含有 '区'
        t.attrib['v'] = new_name[new_name.find(u"区")+1: ]
    elif re.search(re.compile(ur'\('),new_name):
        #如果含有'('
        t.attrib['v'] = new_name[new_name.find(u"(")+1:]
    elif re.search(re.compile(ur'\.|\·|(\s\•\s)'),new_name):
        #如果含有 '.', '·', ' • '
        s = re.search(re.compile(ur'\.|\·|(\s\•\s)'),new_name).group()
        new_name = new_name.split(s)
        t.attrib['v'] = new_name[0]+"."+new_name[-1] #统一为 .
    elif re.search(re.compile(ur'\s'),new_name):
        #如果含有空格
        new_name = new_name.split(u" ")
        t.attrib['v'] = new_name[0]+"."+new_name[-1]
    else:
        #如果是其他情况，删除该tag
        t.attrib['v'] = None
    return t


def update_name(tag):
    st_name = tag.attrib['v']
    if not st_name:
        tag.attrib['v'] = None
    elif re.search(st_type_re1,st_name):
    #1-不含中文和数字
        tag.attrib['v'] = None
    elif st_type_re2.search(st_name):
	#2-有至少一个中文的情况下：
        if st_name.find(u"路") == -1 and st_name.find(u"道") == -1 and \
            st_name.find(u"街") == -1 and st_name.find(u"线") == -1 and \
            st_name.find(u"段") == -1 :
	        #没有有效字
            tag.attrib['v'] = None
        else:
        #有有效字的情况下：
            if st_name[-1] not in expected:
	            #有效字不在末尾的，找到有效字
                for s in expected:
                    if st_name.find(s) != -1:
                        i = st_name.rfind(s) #取最后一个有效字的索引
                        new_name = st_name[:i+1] #例子：Zhao Jia Bing Lu（肇嘉浜路） => Zhao Jia Bing Lu（肇嘉浜路
                        if st_type_re3.search(new_name):
	                        #3-新字符只有中文
                            if new_name.find(u"区") == -1:
	                            #如果新字符不含有 区
                                tag.attrib['v'] = new_name
                            else:
                            #如果新字符含有 区
                                mapping(tag,new_name)
                        else:
                        #新字符内的中文取出，比如：Zhao Jia Bing Lu（肇嘉浜路 => 肇嘉浜路
                            nm = st_type_re2.search(new_name).group()
                            tag.attrib['v'] = nm
            elif st_name[-1] in expected:
	            #有效字在末尾的
                if st_type_re3.search(st_name):
		           #3-原名只有中文，只要检测里面有没有区(或者多个有效的中文字,这里省略了)
                    if st_name.find(u"区") == -1:
	                    #如果没有 区
                        pass
                    else:
	                #如果有 区
                        mapping(tag, st_name)
                else:
                #不只有中文的
                    if st_name.find(u"近茂名路") != -1:
	                    #该数据集的特殊情况
                        tag.attrib['v'] = u"淮海中路"
                    else:
                        mapping(tag, st_name)

    else: #整体可能的其他情况
        tag.attrib['v'] = None



#规整化整个osm文件
def shape_element(elem, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, colon_chars=LOWER_COLON,
                  default_tag_type='regular'):

    nodelem_attrs = {}
    way_attribs = {}
    way_nodes = []
    tags1 = []
    tags2 = []
    tags3 = []

    if elem.tag == 'node':
        elem_attrs = elem.attrib
        for k in node_attr_fields:
            nodelem_attrs[k] = elem_attrs[k]
        for t in elem.iter("tag"):
            if re.search(problem_chars,t.attrib['k']):
                pass
            elif re.search(colon_chars,t.attrib['k']):
                if is_st_name(t):# 含有“:”, 是addr:street的
                    # print "Before: ", t.attrib['v']
                    update_name(t) #更新名字
                    # print "After: ", t.attrib['v']
                    if t.attrib['v']:
                        i = t.attrib['k'].find(':')
                        key = t.attrib['k'][i+1:] #key 一般为street
                        ty = t.attrib['k'][:i] #ty 一般为addr
                        tag1 = {'id':elem_attrs['id'],
                                'key':key,
                                'value':t.attrib['v'],
                                'type':ty}
                        tags1.append(tag1)
                    else:
                        pass
                else:
                    i = t.attrib['k'].find(':')
                    key = t.attrib['k'][i+1:] #key 一般为street
                    ty = t.attrib['k'][:i] #ty 一般为addr
                    tag1 = {'id':elem_attrs['id'],
                            'key':key,
                            'value':t.attrib['v'],
                            'type':ty}
                    tags1.append(tag1)

            else: #不是问题也不是 addr:street的 直接写入
                tag1 = {'id':elem_attrs['id'],
                        'key':t.attrib['k'],
                        'value':t.attrib['v'],
                        'type':default_tag_type}
                tags1.append(tag1)
        return {'node': nodelem_attrs, 'node_tags': tags1}
        del tags1[:]

    elif elem.tag == 'way':
        elem_attrs = elem.attrib
        for k in way_attr_fields:
            way_attribs[k] = elem_attrs[k]

        p = 0
        for t in elem.getchildren():
            if t.tag == 'nd':
                tag2 = {'id':elem_attrs['id'],
                        'node_id':t.attrib['ref'],
                        'position':p}
                p += 1
                tags2.append(tag2)
            elif t.tag =='tag':
                if re.search(problem_chars,t.attrib['k']):
                    pass
                elif re.search(colon_chars,t.attrib['k']):
                    if is_st_name(t):
                        # print "Before: ", t.attrib['v']
                        update_name(t)
                        # print "After: ", t.attrib['v']
                        if t.attrib['v']:
                            i = t.attrib['k'].find(':')
                            key = t.attrib['k'][i+1:]
                            ty = t.attrib['k'][:i]
                            tag3 = {'id':elem_attrs['id'],
                                    'key':key,
                                    'value':t.attrib['v'],
                                    'type':ty}
                            tags3.append(tag3)
                        else:
                            pass
                    else:
                        i = t.attrib['k'].find(':')
                        key = t.attrib['k'][i+1:] #key 一般为street
                        ty = t.attrib['k'][:i] #ty 一般为addr
                        tag3 = {'id':elem_attrs['id'],
                                'key':key,
                                'value':t.attrib['v'],
                                'type':ty}
                        tags3.append(tag3)

                else:
                    tag3 = {'id':elem_attrs['id'],
                            'key':t.attrib['k'],
                            'value':t.attrib['v'],
                            'type':default_tag_type}
                    tags3.append(tag3)
            else:
                pass
        return {'way':way_attribs, 'way_nodes':tags2, 'way_tags':tags3}
        del tags2[:]
        del tags3[:]


def validate_element(document, schema=SCHEMA):
    """如果不匹配schema，显示错误"""
    validator = cerberus.Validator(schema)
    if validator.validate(document) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = u"\n 元素'{0}'出现如下错误:\n{1}"
        error_string = pprint.pformat(errors)
        raise Exception(message_string.format(field, error_string))

class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""
    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""
    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            '''
            if el:
                if validate is True:
                    validate_element(el)
                    '''
            if element.tag == 'node':
                nodes_writer.writerow(el['node'])
                node_tags_writer.writerows(el['node_tags'])
            elif element.tag == 'way':
                ways_writer.writerow(el['way'])
                way_nodes_writer.writerows(el['way_nodes'])
                way_tags_writer.writerows(el['way_tags'])



process_map(OSM_FILE, validate=False)
