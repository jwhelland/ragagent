#!/usr/bin/env bash
set -euo pipefail

NEO4J_URI=${NEO4J_URI:-bolt://neo4j:7687}
NEO4J_USER=${NEO4J_USER:-neo4j}
NEO4J_PASSWORD=${NEO4J_PASSWORD:-neo4j}

echo "Applying schema.cypher to ${NEO4J_URI}..."
cat /init/schema.cypher | cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD"
echo "Neo4j schema initialized."

