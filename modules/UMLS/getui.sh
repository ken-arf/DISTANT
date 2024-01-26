#!/bin/bash

#python search-terms.py -k 3ebafc45-4529-4619-92a3-e3c9432b9e03 -s "T tcell" > tcell.result
#python search-terms.py -k 3ebafc45-4529-4619-92a3-e3c9432b9e03 -s "cytokine" > cytokine.result
#python search-terms.py -k 3ebafc45-4529-4619-92a3-e3c9432b9e03 -s "transcription factor" > transcription_factor.result

#python search-terms.py -k 3ebafc45-4529-4619-92a3-e3c9432b9e03 -s disease > disease.result
#python search-terms.py -k 3ebafc45-4529-4619-92a3-e3c9432b9e03 -s chemicals > chemicals.result

cat chemicals.result| grep -e "MTH" -e "MSH" -e "SNOMED" -e "RXNORM" -e "ICD10" -B 3 | grep ^UI | tr -d "UI: " > chemical.ui
cat disease.result| grep -e "MTH" -e "MSH" -e "SNOMED" -e "RXNORM" -e "ICD10" -B 3 | grep ^UI  | tr -d "UI:" > disease.ui
