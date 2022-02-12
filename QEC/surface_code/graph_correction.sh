#!/bin/bash
sed -i 's/weight/label/g' primal_de.dot
sed -i 's/weight/label/g' dual_de.dot
dot -Tsvg primal_de.dot > primal_de.svg
dot -Tsvg dual_de.dot > dual_de.svg
