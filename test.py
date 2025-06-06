class Robot:
     Directions=["NORTH","EAST","SOUTH","WEST"]
     def __init__(self):
       self.x =None
       self.y =None
       self.facing=None
       self.placed= False


     def Place(self,x,y,facing):
       if facing in Robot.Directions:
          self.x=x
          self.y=y
          self.facing=facing
          self.placed=True

          print(f"Robot placed at ({self.x},{self.y}) facing {self.facing}")
     def Move(self):
        if not self.placed:
           return 
        if self.facing=="NORTH":
            self.y+=1
        if self.facing=="SOUTH":
            self.y-=1
        if self.facing=="EAST":
            self.x+=1
        if self.facing=="WEST":
            self.x-=1
        print(f"Robot moved to ({self.x},{self.y})")
     def left(self):
        if not self.placed:
           return
        
        idx=self.Directions.index(self.facing)
        self.facing=self.Directions[(idx-1)%4]
        print(f"Robot turned left and is now facing {self.facing}")
     
     def right(self):
         if not self.placed:
           return
         idx=self.Directions.index(self.facing)
         self.facing=self.Directions[(idx+1)%4]
         print(f"Robot turned Right and is now facing {self.facing}")
     
     def report(self):
        if self.placed:
           print(f"Robot is at ({self.x},{self.y}) facing {self.facing}")
     
def run():
       robot=Robot()

       while not robot.placed:
          start_input=input("Enter the command:")
          if start_input.startswith("PLACE"):
             try:
                _,args=start_input.split()
                print(args)
                x,y,facing=args.split(",")
                print(x,y,facing)
                # if not x.isdigit() or not y.isdigit():
                #     print("Invalid Place command")
                robot.Place(int(x),int(y),facing.upper())
             except Exception as e:
                print(e)
                print("Invalid Place command")
                print("Please enter the command in the format: PLACE X,Y,FACING")
             while True:
                  command=input("Enter the command:").strip().upper()
                  if command=="MOVE":
                    robot.Move()
                  elif command=="LEFT":
                    robot.left()
                  elif command=="RIGHT":
                    robot.right()
                  elif command=="REPORT":
                    robot.report()
                  elif command=="EXIT":
                    print("Exiting the program")
                    break
                  else:
                    print("Invalid command")

if __name__=="__main__":
       run()

                