# Notes

pg_dump the snowflake3 schema into tar file:  
``` bash
$ sudo su - postgres
$ source .bashrc
$ cd /dias_repo/datasets/mpdp/
$ pg_dump -Ft -v snowflake3 > snowflake.tar
```

*Notes: su postgres does not have access to backup directory*
