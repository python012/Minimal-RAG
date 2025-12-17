#!/bin/bash
# Load JSON recipe files into vector database
echo "Loading JSON recipes from data/recipes/..."
python data_loader.py --input ./data/recipes/ --pattern *.json
echo ""
echo "Done!"
read -p "Press Enter to continue..."