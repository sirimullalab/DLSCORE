# Setup virtual environment

virtualenv -p $(which python3.6) .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download the network files

if [ ! -d ./networks ]; then
  mkdir networks/
  curl -o general.tar.gz https://drugdiscovery.utep.edu/download.php?file=general.tar.gz
  curl -o refined.tar.gz https://drugdiscovery.utep.edu/download.php?file=refined.tar.gz

  tar -xvf general.tar.gz && mv general networks/ && rm -f general.tar.gz
  tar -xvf refined.tar.gz && mv refined networks/ && rm -f refined.tar.gz
fi
