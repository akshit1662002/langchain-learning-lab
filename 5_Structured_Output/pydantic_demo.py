from pydantic import BaseModel , EmailStr , Field
from typing import Optional


class Student(BaseModel):
    name : str = 'akshit'
    age : Optional[int] = None
    email :  EmailStr
    cgpa : float = Field(gt = 0 ,lt = 10 , default=0 , description='A decimal value represent cgpa of the student' , ) #constrain

new_student = {'age' : 32 , 'email' : 'abs@gmail.com'  }

student = Student(**new_student)

print(student)
print(student.name) 

#convert it into dict
student_dict = dict(student)

print(student_dict)

#convert it into json 
student_json = student.model_dump_json()
print(student_json)