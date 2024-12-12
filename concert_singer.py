import random
import string

def random_string(length=10):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))

def generate_stadium_row(stadium_id):
    location = random_string(10)
    name = random_string(15)
    capacity = random.randint(2000, 60000)
    highest = random.randint(1000, capacity)
    lowest = random.randint(100, highest)
    average = random.randint(lowest, highest)
    return f"INSERT INTO stadium VALUES ({stadium_id}, '{location}', '{name}', {capacity}, {highest}, {lowest}, {average});"

def generate_singer_row(singer_id):
    name = random_string(10)
    country = random.choice(['United States', 'France', 'Netherlands', 'China'])
    song_name = random_string(8)
    song_release_year = str(random.randint(1990, 2023))
    age = random.randint(20, 60)
    is_male = 'TRUE' if random.choice([True, False]) else 'FALSE'
    return f"INSERT INTO singer VALUES ({singer_id}, '{name}', '{country}', '{song_name}', '{song_release_year}', {age}, {is_male});"

def generate_concert_row(concert_id, stadium_id):
    concert_name = random_string(12)
    theme = random_string(15)
    year = str(random.randint(2010, 2023))
    return f"INSERT INTO concert VALUES ({concert_id}, '{concert_name}', '{theme}', {stadium_id}, '{year}');"

def generate_singer_in_concert_row(concert_id, singer_id):
    return f"INSERT INTO singer_in_concert VALUES ({concert_id}, {singer_id});"

# Example usage:
for i in range(10, 200):  # Generate 10 rows
    print(generate_stadium_row(i))
    print(generate_singer_row(i))
    print(generate_concert_row(i, i))
    print(generate_singer_in_concert_row(i, i))
