import importlib

#  Driver code
if __name__ == "__main__":

    my_module = importlib.import_module("preprocess.annotation.defines.immunology.entity_define")
    print(my_module.get_entity_defines)


