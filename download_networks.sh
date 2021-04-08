if [ ! -d ./networks ]; then
  mkdir networks/
  # Download top 10 network weights
  curl -o general.tar.gz https://drugdiscovery.utep.edu/download.php?file=general-10.tar.gz
  curl -o refined.tar.gz https://drugdiscovery.utep.edu/download.php?file=refined-10.tar.gz
  
  # Download all 100 network weights
  # curl -o general.tar.gz https://drugdiscovery.utep.edu/download.php?file=general-10.tar.gz
  # curl -o refined.tar.gz https://drugdiscovery.utep.edu/download.php?file=refined-10.tar.gz
  tar -xvf general.tar.gz && mv general networks/ && rm -f general.tar.gz
  tar -xvf refined.tar.gz && mv refined networks/ && rm -f refined.tar.gz
fi
