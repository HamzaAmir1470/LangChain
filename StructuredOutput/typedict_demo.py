from typing import TypedDict


class Person(TypedDict):
    name: str
    age: int
    email: str


new_person: Person = {
    "name": "Hamza Amir",
    "age": 30,
    "email": "hamzaamir9733@gmail.com",
}

print(new_person)
