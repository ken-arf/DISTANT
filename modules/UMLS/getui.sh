#!/bin/bash

#python search-terms.py -k 3ebafc45-4529-4619-92a3-e3c9432b9e03 -s disease > disease.result
#python search-terms.py -k 3ebafc45-4529-4619-92a3-e3c9432b9e03 -s chemicals > chemicals.result

cat chemicals.result| grep -e "MSH" -e "SNOMED" -e "RXNORM" -B 3 | grep ^UI | tr -d "UI: " > chemical.ui
cat disease.result| grep -e "MSH" -e "SNOMED" -e "RXNORM" -B 3 | grep ^UI  | tr -d "UI:" > disease.ui
