#!/bin/bash
# Clear ChromaDB vector database
echo "Clearing database..."
python data_loader.py --clear
echo ""
echo "Done!"
read -p "Press Enter to continue..."