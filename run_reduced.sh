# This is the main script to run the Federated Learning Setup
# It executes the Flower Server (server.py) and multiple clients with
# the script client.py and argument --silo=PATH
#
# If is desired to run the FL with the EFC additional feature, 
# keep the --with_EFC argument to Python scripts
#
#!/bin/bash

echo "Starting Federated Learning server using Flower"

python server.py --with_efc &
sleep 10  # Sleep for 10s to give the server enough time to start

for filename in ./reduced_datasets/*.csv.gz; do
    echo "Starting client with silo $filename"
    python client.py --silo=${filename} --with_efc &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
