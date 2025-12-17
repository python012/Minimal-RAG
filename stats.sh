#!/bin/bash
# Show database statistics
echo "Database Statistics:"
echo ""
python data_loader.py --stats
echo ""
read -p "Press Enter to continue..."