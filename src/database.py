import sqlite3

db_path = 'barc.db'


def count():
    cursor = sqlite3.connect(db_path).cursor()
    return cursor.execute('SELECT count(*) FROM images').fetchone()[0]

def count_by_category():
    cursor = sqlite3.connect(db_path).cursor()
    return cursor.execute('''
        SELECT c.object, count(*)
        FROM categories as c
        INNER JOIN images_categories AS i ON i.category_id = c.id
        GROUP BY c.object
    ''').fetchall()

def images():
    cursor = sqlite3.connect(db_path).cursor()
    query = 'SELECT id FROM images' 
    result = cursor.execute(query).fetchall()
    return [ row[0] for row in result ]

def categories():
    cursor = sqlite3.connect(db_path).cursor()
    query = 'SELECT id, object FROM categories ORDER BY object'
    categories = cursor.execute(query).fetchall()
    ids = [ row[0] for row in categories ]
    labels = [ row[0] for row in categories ]
    return ids, labels

def cat_for_image(image_id, category_ids):
    cursor = sqlite3.connect(db_path).cursor()
    query = 'SELECT category_id FROM images_categories WHERE image_id == "%s"'
    result = cursor.execute(query % image_id).fetchall()
    image_cats = [ row[0] for row in result ]
    return [ 1 if cat in image_cats else 0 for cat in category_ids ]


