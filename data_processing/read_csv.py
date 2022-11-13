import os
from io import StringIO
import csv
import sys
csv.field_size_limit(sys.maxsize)

LOCATION = "/scr/dlvp_local_data/code_files/bugzilla_snykio_V3/all_neo4jcsv"
# LOCATION = "/home/ding/dlvp/dl-vulnerability-detection/data/commits/code/test/parsed"
# for direc in os.listdir(f"{LOCATION}"):
#     all_edges = ":START_ID,:END_ID,:TYPE\n"
#     for file in os.listdir(f"{LOCATION}/"+direc):
#         if file.startswith("edges") and file.endswith("data.csv"):
#             with open(f"{LOCATION}/"+direc+"/"+file, "r") as f:
#                 all_edges += f.read()
#     with open(f"{LOCATION}/"+direc+"/edges.csv", "w+") as f:
#         f.write(all_edges)
types = set()
for direc in os.listdir(f"{LOCATION}"):
    header = ':ID,:LABEL,CODE:string,LINE_NUMBER:int'+'\n'
    for file in os.listdir(f"{LOCATION}/"+direc):
        if file.startswith("nodes") and file.endswith("header.csv"):
            node_type = file.replace("header.csv", "data.csv")
            with open(f"{LOCATION}/"+direc+"/"+file, "r") as f:
                result = f.read()
            with open(f"{LOCATION}/"+direc+"/"+node_type, "r") as f:
                result += f.read()
            f = StringIO(result)
            reader = csv.DictReader(f)
            for row in reader:
                id = row[':ID']
                label = row[':LABEL']
                if label == "CALL":
                    if "<operator>." in row['NAME:string']:
                        label += ":" + row['NAME:string'].replace("<operator>.", "")
                if label == "CONTROL_STRUCTURE":
                    label += ":" + row['PARSER_TYPE_NAME:string']
                if label == "JUMP_TARGET":
                    label += ":" + row['PARSER_TYPE_NAME:string']
                if label == "METHOD":
                    if "<operator>." in row['NAME:string']:
                        label += ":" + row['NAME:string'].replace("<operator>.", "")
                types.add(label)
                try:
                    code = " ".join(row['CODE:string'].split())
                except KeyError:
                    code = ''
                try:
                    line_no = row['LINE_NUMBER:int']
                except KeyError:
                    line_no = ''
                header += ','.join([id,label,code,line_no]) + '\n'
    with open(f"{LOCATION}/"+direc+"/nodes.csv", "w+") as f:
        f.write(header)
dic = {}
for i, t in enumerate(types):
    dic[t] = i+1
print(dic)