#!/bin/bash

OLDIFS=$IFS
IFS=$(echo -en "\n\b")

for fn in $(ls *.txt); do
  line=$(cat "$fn" | grep '[01]')
  label=${line:0:1}
  echo $label
  new_fn=${label}_$fn
  echo $new_fn
  mv ${fn} $new_fn
done
IFS=$OLDIFS

echo 'done'
