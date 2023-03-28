import pymysql

USERNAME = 'root'
PASSWORD = '314zktla'
DBNAME = 'nas_db'

# Connect to the MySQL server
conn = pymysql.connect(host='localhost', user=USERNAME, password=PASSWORD)

# Create a cursor object
cur = conn.cursor()

# Execute a SQL query to retrieve a list of all databases on the server
cur.execute("SHOW DATABASES")

# Fetch all the rows in the result set
databases = cur.fetchall()

# Check if the database we're looking for is in the list
if (DBNAME,) in databases:
    print("The database exists.")
    conn.select_db(DBNAME)
else:
    print("The database does not exist.")
    cur.execute("CREATE DATABASE " + DBNAME)
    conn.select_db(DBNAME)

# Execute the SQL statement to create the table

sql = '''SELECT COUNT(*)
         FROM information_schema.tables
         WHERE table_name = 'blocks' AND table_schema = 'nas_db' '''

# Execute the SQL statement
cur.execute(sql)

# Get the query result
result = cur.fetchone()

# Check if the table exists
if result[0] == 1:
    print("The table exists.")
    cur.execute("DROP TABLE IF EXISTS blocks")

sql = '''CREATE TABLE blocks (
        id INT AUTO_INCREMENT PRIMARY KEY,
        first_conv VARCHAR(50) NOT NULL,
        first_group INT(11) NOT NULL,
        second_conv VARCHAR(50) NOT NULL,
        second_kernel VARCHAR(50) NOT NULL,
        SE_Bool VARCHAR(10) NOT NULL,
        SE_ratio INT(11) NOT NULL,
        activation VARCHAR(10) NOT NULL,
        expansion_ratio INT(11) NOT NULL,
        priority INT(11) NOT NULL
    )'''
cur.execute(sql)
print("The table is empty.")
# how to insert entries into the table?
sql = "INSERT INTO blocks (first_conv, first_group, second_conv, second_kernel, SE_Bool, SE_ratio, activation, expansion_ratio, priority) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
values = [
        ("P", 1, "D", '3', "yes", 16, "R", 1, 1),
        ("G", 2, "D", '3', "no", 16, "R", 1, 2),
        ("G", 3, "D", '3', "no", 16, "R", 1, 3),
        ("G", 3, "M", '3_5', "yes", 16, "S", 1, 4),
        ("P", 3, "M", '3_5_7', "yes", 16, "S", 2, 5),
        ("P", 3, "M", '3_5', "no", 16, "S", 2, 6),
        ("G", 3, "D", '5', "yes", 16, "S", 2, 7),
        ("P", 1, "D", '5', "yes", 16, "S", 2, 8),
        ("P", 3, "M", '5_7', "yes", 16, "R", 2, 9),
          ]
cur.executemany(sql, values)
# Commit the transaction



conn.commit()
cur.close()
conn.close()
