import xml.etree.cElementTree as ET # Use cElementTree or lxml if too slow


OSM_FILE = "shanghai_major.osm"  # Replace this with your osm file
SAMPLE_FILE = "sample_small.osm"
tags=('node', 'way', 'relation')

k = 29  # Parameter: take every k-th top level element


def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag
    Reference:
    http://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
    """
    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem  #<generator object get_element at 0x104feaaa0>
            root.clear() #清除root里<osm> 的子节点




with open(SAMPLE_FILE, 'wb') as output:
    output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write('<osm>\n')

    # Write every kth top level element

    for i, element in enumerate(get_element(OSM_FILE)):
        #print ET.tostring(element, encoding='utf-8')
        if i % k == 0:
            output.write(ET.tostring(element, encoding='utf-8')) # xml.etree.ElementTree.tostring(element, encoding="us-ascii", method="xml")

    output.write('</osm>')
