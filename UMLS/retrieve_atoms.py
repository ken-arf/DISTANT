#This program returns atoms from CUI or code input.
#If using a code, the source vocabulary must be specified.

import requests
import argparse

import pdb

parser = argparse.ArgumentParser(description='process user given parameters')
parser.add_argument('-k', '--apikey', required = False, dest = 'apikey', default='3ebafc45-4529-4619-92a3-e3c9432b9e03', help = 'enter api key from your UTS Profile')
parser.add_argument('-v', '--version', required =  False, dest='version', default='current', help = 'enter version example-2015AA')
parser.add_argument('-s', '--source', required =  False, dest='source', help = 'enter source name if known')
parser.add_argument('-l', '--language', required = False, dest='language', default = 'ENG', help = 'enter source name if known')
parser.add_argument("-i", "--identifier", required =  True, dest="identifier", help = "enter identifier example-C0018787")

args = parser.parse_args()

apikey = args.apikey
version = args.version
source = args.source
language = args.language
identifier = args.identifier

uri = 'https://uts-ws.nlm.nih.gov'
page = 0

try:
   source
except NameError:
   source = None

#If we don't specify a source vocabulary, it's assumed we're using UMLS CUIs.
if source is None:
    content_endpoint = '/rest/content/'+str(version)+'/CUI/'+str(identifier)
else:
    content_endpoint = '/rest/content/'+str(version)+'/source/'+str(source)+'/'+str(identifier)
                
query = {'apiKey':apikey, 'language':language}
print(uri+content_endpoint)
print(query)

r = requests.get(uri+content_endpoint, params=query)
r.encoding = 'utf-8'
            
if r.status_code != 200:
    if source is None:
        raise Exception('Search term ' + "'" + str(identifier) + "'" + ' not found')
    else:
        raise Exception('Search term ' + "'" + str(identifier) + "'" + 'or source ' + "'" + str(source) + "'" + ' not found')


            
items  = r.json()
print(items)


jsonData = items['result']
            
Atoms = jsonData['atoms']

#The below uses the atoms' URI value from above as the starting URI.

AUI_list = []
try:   
    while True:
        page += 1
        atom_query = {'apiKey':apikey, 'pageNumber':page, 'language':language}
        a = requests.get(Atoms, params=atom_query)
        a.encoding = 'utf-8'
        
        if a.status_code != 200:
            break

        all_atoms = a.json()
        jsonAtoms = all_atoms['result']
    
        for atom in jsonAtoms:
            print('Name: ' + atom['name'])
            print('CUI: ' + jsonData['ui'])
            print('AUI: ' + atom['ui'])
            print('Term Type: ' + atom['termType'])
            print('Code: ' + atom['code'])
            print('Source Vocabulary: ' + atom['rootSource'])
            print('\n')
            AUI_list.append(atom['ui'])
except Exception as except_error:
    print(except_error)

print(AUI_list)

for AUI in AUI_list:

    print(AUI)
    print('-' * 10)
    content_endpoint = f"/content/current/AUI/{AUI}/descendants"

    query = {'apiKey':apikey}
    r = requests.get(uri+content_endpoint, params=query)
    r.encoding = 'utf-8'
    
    if r.status_code != 200:
        #break
        continue

    all_atoms = r.json()
    jsonAtoms = all_atoms['result']

    for atom in jsonAtoms:
        print('Name: ' + atom['name'])
        print('CUI: ' + jsonData['ui'])
        print('AUI: ' + atom['ui'])
