from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "Hamza Amir"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, description="CGPA must be between 0 and 10")


new_student = Student(
    name="Hamza Amir", age=30, email="hamzaamir9733@gmail.com", cgpa=9.5
)

student = Student(**new_student.model_dump())

print(student)
