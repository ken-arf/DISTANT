#This script displays hierarchical relationships for an identifier (code).
#Source vocabulary is required.
#Only one type of hierarchical relationship can be searched for at a time - such as children, parents, descendents, or ancestors.

import requests
import argparse


def get_source_atoms(apikey, source, identifier, operation):

    version = "current"
    uri = "https://uts-ws.nlm.nih.gov"
    content_endpoint = "/rest/content/"+version+"/source/"+source+"/"+identifier+"/"+operation

    all_items = []

    pageNumber=0

    try:
        while True:
            pageNumber += 1
            query = {'apiKey':apikey,'pageNumber':pageNumber}
            r = requests.get(uri+content_endpoint,params=query)
            r.encoding = 'utf-8'
            items  = r.json()
            
            if r.status_code != 200:
                if pageNumber == 1:
                    print('No results found.'+'\n')
                    break
                else:
                    break
                
            print("Results for page " + str(pageNumber)+"\n")
            
            for result in items["result"]:
                ui = name = source = ""
                try:
                    print("ui: " + result["ui"])
                    ui = result["ui"]
                except:
                    NameError
                try:
                    print("uri: " + result["uri"])
                except:
                    NameError
                try:
                    print("name: " + result["name"])
                    name = result["name"]
                except:
                    NameError
                try:
                    print("Source Vocabulary: " + result["rootSource"])
                    source = result["rootSource"]
                except:
                    NameError
                print("\n")

                all_items.append((ui, name, source))
        print("*********")
    except Exception as except_error:
        print(except_error)

    return all_items
