#!/bin/bash

curl -H "Content-Type: application/json" -X POST http://localhost:9200/_bulk?pretty --data-binary @database/abstract.jsonl
curl -H "Content-Type: application/json" -X POST http://localhost:9200/_bulk?pretty --data-binary @database/entity.jsonl

