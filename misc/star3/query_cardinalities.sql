SELECT relname, relkind, reltuples, relpages
FROM pg_class
WHERE relname LIKE 't\_%' AND relkind = 'r';