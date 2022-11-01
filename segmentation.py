from PyQt5 import QtCore, QtGui, QtWidgets  # core Qt functionality
from PyQt5 import QtOpenGL  # provides QGLWidget, a special OpenGL QWidget
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtCore import Qt
import math
from scipy.interpolate import RegularGridInterpolator
import qimage2ndarray
 
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import OpenGL.GL as gl  # python wrapping of OpenGL
from OpenGL import GLU  # OpenGL Utility Library, extends OpenGL functionality
from OpenGL.arrays import vbo
import numpy as np

import sys  # we'll need this later to run our Qt application
import time 

"""
Notes:

Coordinate Systems:
    The viewing plane is in dimensions self.windowWidth by self.windowHeight, with point (0,0) in the upper left corner. 
    Variable self.c is the center of the plane
    
    The global coordinates refer to where we are in 3D space. Coordinate (0,0,0) is in the upper left corner. 
    
    OpenGL coordinates refers to the coordinates in our 3D window. 
    The global coordinates are scaled such that the width and height of the image volume is transformed to a length of 1
    Center is at (0.5,0.5,0.5)
    
Data structures:
    self.annotation: 
        in class GLWidget  (class instance glWidget)
        stores all the annotations in a data structure of nested lists
        Each branch is represented as a list of annotations
        Each annotation is represented as a list where index 0 stores the type of annotation (junction or polygon) and index 1 stores the coordinates
        Each coordinate is an array of length 3 that stores x,y,z coordinates
    
    self.skeleton:
        in class GLWidget  (class instance glWidget)
        stores all the points that make up the skeletons
        Each branch is represented as a list of points
        Each point is an array of length 3 that stores x,y,z coordinates
        
    self.polygon_colors
        in class GLWidget (class instance glWidget)
        stores all the colors for each branch
        The color stored in self.polygon_colors[i] is the color for branch i
        For example, the color in index 3 is for branch stored in self.annotations[3]
    
    self.polygon_planes
        in class GraphicsView (class instance view)
        stores all the planes for each annotation
        
        Each branch is represented as a list of planes
        Each plane is stored as a list of v1,v2,n, and c. 
        When a user finishes annotating, the plane is saved so the user can go back to the plane
        The index of the plane matches the index of the annotation in self.glWidget.annotations
        For example, if the annotation is in index 1 of branch 2, then the plane for that annotation is saved in self.polygon_planes[2][1]
        
    self.hierarchy
        in class MainWindow
        stores the hierarchy - the relationship between parents and children branches
        Each branch is represented as a list of children branches
        For example, if self.hierarchy = [[1,2][3]], then branch 0 has children 1 and 2 and branch 1 has child 3. 
        These numbers refer to the index of the branch stored in self.annotation / self.glWidget.annotation
     
The viewing plane is defined by vectors v1 and v2, the cross product n defining the normal vector, and point c defining the center of the plane
        
"""


"""
Where 3D annotations, viewing plane, and image volume are displayed.
Reffered to as "OpenGL window" or "3D window"
"""
class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)
        # dimensions of image volume
        self.volumeWidth = 400
        self.volumeHeight = 400
        self.volumeLength = 250
        # width of viewing plane after scaled
        self.width_Scaled = 800
        self.height_Scaled = 800
        # vectors and points of viewing plane
        self.v1 = np.array([1, 0, 0])
        self.v2 = np.array([0, 1, 0])
        self.n = np.cross(self.v1, self.v2)
        self.c = np.array([self.volumeWidth / 2, self.volumeHeight / 2, 0]) #center of viewing plane
        self.p = [np.array([0, 0, 0])]
        self.corners = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0]), np.array([0, 1, 0])]
        self.quadric = GLU.gluNewQuadric()
        # variables for rotating and translating in 3d window
        self.startX = 0
        self.startY = 0
        self.startV1 = np.array([0, 0, 0])
        self.startV2 = np.array([0, 0, 0])
        self.startN = np.array([0, 0, 0])
        self.rotX = 0.0
        self.rotY = 0.0
        self.rotZ = 0.0
        self.scaleZ = 0.0
        self.start_rotX = 0
        self.start_rotY = 0
        self.start_rotZ = 0
        self.scale = 1
        self.start_pan = np.array([0, 0, 0])
        self.pan = np.array([0, 0, 0])
        # determines what is being rendered in the 3d window
        self.loadGT = False # has ground truth been loaded?
        self.showGT = False # does the user want to see the ground truth rendering?
        self.renderAll = True # show polygons and junction points
        self.values = [] # stores the pixels that the biologist painted in the ground truth
        # lists and variables for annotations
        self.annotations = []
        self.polygon_colors = []
        self.skeleton = []
        self.currBranch = 0
        self.currPolygon = 0

    """
    Called once to set up the OpenGL rendering context before resize or paint are ever called
    """
    def initializeGL(self):
        self.qglClearColor(QColor(0, 0, 0))  # initialize the screen to black
        gl.glEnable(gl.GL_DEPTH_TEST)  # enable depth testing
        self.initGeometry()

    """
    Given 3d coordinates
    Transforms given point to OpenGL coordinate system. Sets self.c to transformed point
    """
    def setC(self, c):
        self.c = np.array([c[0] / self.volumeWidth, c[1] / self.volumeWidth, (c[2] / self.volumeWidth) + (1/2 - (self.volumeLength/(self.volumeWidth*2)))])

    """
    When user presses the mouse in 3d window. 
    Sets start values used for determining how much the user wants to translate or rotate
    """
    def mousePressEvent(self, event):
        posMouse = event.pos()
        if self.rect().contains(posMouse):
            if event.button() == Qt.LeftButton:
                self.startX = event.x()
                self.startY = event.y()
                self.start_rotX = self.rotX
                self.start_rotY = self.rotY
                self.start_rotZ = self.rotZ
            if event.button() == Qt.RightButton:
                self.startX = event.x()
                self.startY = event.y()
                self.start_pan = self.pan

    """
    When the user moves their mouse in the window with the left button, the scene is rotated by updating self.rotX and self.rotZ
    When the user moves their mouse in the window with the right button, the scene is translated by updating self.pan
    Variables self.rotX, self.rotZ, and self.pan are used the transform the scene in function paintGL
    """
    def mouseMoveEvent(self, event):
        posMouse = event.pos()
        # how far the mouse has been dragged in the x and y direction
        x_disp = self.startX - event.x()
        y_disp = self.startY - event.y()
        if self.rect().contains(posMouse):

            if event.buttons() & Qt.LeftButton:
                self.rotX = self.start_rotX + y_disp / 3
                self.rotZ = self.start_rotZ + x_disp / 3
                glutPostRedisplay()

            if event.buttons() & Qt.RightButton:
                self.pan = self.pan - np.array([x_disp/self.volumeLength,-y_disp/self.volumeLength,0])
                self.startX = event.x()
                self.startY = event.y()
                glutPostRedisplay()

    """
    When the user scrolls in the window, will zoom in or out
    """
    def wheelEvent(self, event):
        posMouse = event.pos()
        if self.rect().contains(posMouse):
            numSteps = event.angleDelta().y() / 120
            self.scale = self.scale - numSteps/10
            glutPostRedisplay()

    """
    Called once when the window is created and whenever the window is resized to set up the OpenGL viewport and projection
    """
    def resizeGL(self, width, height):
        gl.glViewport(0, 0, width, height)  # specifies which portion of the screen of the window is used for drawing
        gl.glMatrixMode(GL_PROJECTION)  # projection transformation, defines the viewing volume
        gl.glLoadIdentity()  # first load s the identity matrix
        self.aspect = width / float(height)

        gluPerspective(-45.0, self.aspect, 1, 100)  # viewing frustrum w/ Perspective
        gl.glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    """
    Given:
        x,y,z: coordinates in OpenGL coordinate system
        color: array storing r,g,b values 
    Renders a cube of side length 2/self.volumeLength at given coordinates in given color
    A cube of this side length is the same as a cube with side length of 2 in our global coordinate system. 
    """
    def renderCube(self,x,y,z,color):
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        c = 1/self.volumeLength

        glBegin(GL_QUADS)
        gl.glColor3f(color[0]*0.7, color[1]*0.7, color[2]*0.7)
        glVertex3f(x - c, y - c, z-c)
        glVertex3f(x + c, y - c, z-c),
        glVertex3f(x + c, y + c, z-c),
        glVertex3f(x-c, y + c, z-c),
        glEnd()

        glBegin(GL_QUADS)
        gl.glColor3f(color[0] * 0.7, color[1] * 0.7, color[2] * 0.7)
        glVertex3f(x - c, y + c, z-c),
        glVertex3f(x+c, y+c, z-c),
        glVertex3f(x+c, y+c, z+c)
        glVertex3f(x - c, y+c, z+c)
        glEnd()

        glBegin(GL_QUADS)
        gl.glColor3f(color[0] * 0.5, color[1] * 0.5, color[2] * 0.5)
        glVertex3f(x + c, y - c, z - c),
        glVertex3f(x - c, y - c, z-c),
        glVertex3f(x - c, y - c, z+c)
        glVertex3f(x+c, y- c, z+c)
        glEnd()

        glBegin(GL_QUADS)
        gl.glColor3f(color[0], color[1], color[2])
        glVertex3f(x + c, y + c, z-c),
        glVertex3f(x + c, y - c, z-c),
        glVertex3f(x + c, y - c, z + c)
        glVertex3f(x+c, y+c, z+c)
        glEnd()

        glBegin(GL_QUADS)
        gl.glColor3f(color[0] * 0.5, color[1] * 0.5, color[2] * 0.5)
        glVertex3f(x-c, y-c, z-c),
        glVertex3f(x-c, y+c, z-c)
        glVertex3f(x-c, y+c, z+c)
        glVertex3f(x-c, y-c, z+c)
        glEnd()

        glBegin(GL_QUADS)
        gl.glColor3f(color[0] * 0.7, color[1] * 0.7, color[2] * 0.7)
        glVertex3f(x-c, y+c, z+c)
        glVertex3f(x+c, y+c, z+c)
        glVertex3f(x+c, y- c, z+c)
        glVertex3f(x-c, y- c, z+c)
        glEnd()


    """
    Renders the ground truth. The ground truth is what the biologist annotated. Root is indicated by painted pixels
    Called once the user loads a folder of ground truth images and clicks the checkbox "ground truth"
    """
    def renderGroundTruth(self):
        length = self.volumeLength
        self.displayList = glGenLists(1) # display list
        glNewList(self.displayList, GL_COMPILE)
        values = self.values # 3D matrix storing 0 and 1. 1 indicates there is a painted pixel
        for z in range(length-1): # loop through each image in our ground truth data
            A = values[z] # matrix of current image
            #use sparse matrices
            yCoord = A.nonzero()[0] # y coordinates that have a 1
            xCoord = A.nonzero()[1] # x coordinates that have a 1
            x = xCoord/self.volumeWidth # transform to OpenGL coordinate system
            y = yCoord/self.volumeWidth  # transform to OpenGL coordinate system
            z = z/self.volumeWidth + (1/2 - (self.volumeLength/(self.volumeWidth*2)))  # transform to OpenGL coordinate system
            for i in range(len(xCoord)): #for every coordinate that has a 1
                self.renderCube(x[i], y[i], z,[1,0,0]) #render cube in red
        glEndList()

    """
    Render viewing plane.
    Corners defined in array self.corners
    """
    def renderPlane(self):

        p1 = self.corners[0]
        p2 = self.corners[1]
        p3 = self.corners[2]
        p4 = self.corners[3]

        glColor3f(1.0, 0.0, 0.0)
        glLineWidth(5.0)
        glBegin(GL_LINES)
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p2[0], p2[1], p2[2])
        glEnd()

        glColor3f(0.0, 1.0, 0.0)
        glLineWidth(5.0)
        glBegin(GL_LINES)
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p4[0], p4[1], p4[2])
        glEnd()

        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(5.0)
        glBegin(GL_LINES)
        glVertex3f(p3[0], p3[1], p3[2])
        glVertex3f(p4[0], p4[1], p4[2])
        glEnd()

        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(5.0)
        glBegin(GL_LINES)
        glVertex3f(p2[0], p2[1], p2[2])
        glVertex3f(p3[0], p3[1], p3[2])
        glEnd()

    """
    Given a direction and color, renders a cylinder representing vectors of viewing plane
    """
    def renderArrow(self, dir, color):
        # Compute the angle-axis rotation require to orient the vector along dir:
        glLineWidth(1.0)
        up_vec = np.array([0.0, 0.0, 1.0])
        axis = np.cross(up_vec, dir)
        trip_prod = np.linalg.det(np.dstack((up_vec, dir, axis)))
        if trip_prod > 0:
            angle = np.arccos(np.dot(up_vec, dir))
        else:
            angle = 2 * np.pi - np.arccos(np.dot(up_vec, dir))

        # Draw the shaft using a cylinder:
        gl.glPushMatrix()
        gl.glColor3f(*color)
        gl.glTranslatef(*self.c)
        gl.glRotate((180.0 / np.pi) * angle, *axis)
        gluCylinder(self.quadric, 0.005, 0.005, 0.1, 100, 10)
        gl.glPopMatrix()

    """
    Renders annotations saved in self.annotations
    See data structures notes at the top for more information about self.annotations and self.polygon_colors
    """
    def renderAnnotations(self):
        if len(self.annotations)>0:
            c = 0
            for branch in self.annotations:
                num = 1
                for annotation in branch:
                    if annotation[0]=="polygon":
                        polygon = annotation[1]
                        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                        color = self.polygon_colors[c]
                        glColor3f(color.red()/255, color.green()/255, color.blue()/255)
                        for i in range(1,len(polygon)-1):
                            glBegin(GL_TRIANGLES)
                            glVertex3f(polygon[0][0], polygon[0][1], polygon[0][2])
                            glVertex3f(polygon[i][0], polygon[i][1], polygon[i][2])
                            glVertex3f(polygon[i+1][0], polygon[i+1][1], polygon[i+1][2])
                            glEnd()
                        glRasterPos3f(polygon[0][0]-0.01, polygon[0][1]-0.01, polygon[0][2]-0.01)

                    if annotation[0] == "junction":
                        junction = annotation[1]
                        color = self.polygon_colors[c]
                        self.renderCube(junction[0],junction[1],junction[2],[color.red()/255, color.green()/255, color.blue()/255])
                        glRasterPos3f(junction[0] - 0.01, junction[1] - 0.01, junction[2] - 0.01)
                        
                    if annotation[0] == "tracking":
                        tracking = annotation[1]
                        color = self.polygon_colors[c]
                        self.renderCube(tracking[0],tracking[1],tracking[2],[color.red()/255, color.green()/255, color.blue()/255])
                        glRasterPos3f(tracking[0] - 0.01, tracking[1] - 0.01, tracking[2] - 0.01)

                    # renders text representing branch and index of annotation
                    char = c+1
                    if char < 10:
                        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(str(char)))
                    elif char < 100:
                        ones = char%10
                        tens = int(char/10)
                        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(str(tens)))
                        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(str(ones)))
                    else:
                        ones = char%10
                        tens = char%100 - ones
                        hundreds = int(char/100)
                        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(str(hundreds)))
                        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(str(tens)))
                        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(str(ones)))

                    glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord("."))
                    if num < 10:
                        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(str(num)))
                    elif num < 100:
                        ones = num%10
                        tens = int(num/10)
                        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(str(tens)))
                        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(str(ones)))
                    else:
                        ones = num%10
                        tens = num%100 - ones
                        hundreds = int(num/100)
                        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(str(hundreds)))
                        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(str(tens)))
                        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(str(ones)))
                    num += 1
                c += 1

    """
        Renders skeleton saved in self.skeleton
        See data structures notes at the top for more information about self.skeleton and self.polygon_colors
    """
    def renderSkeleton(self):
        c = 0
        for branch in self.skeleton:
            if len(branch)>1:
                color = self.polygon_colors[c]
                glColor3f(color.red() / 255, color.green() / 255, color.blue() / 255)
                for i in range(len(branch)-1):
                    p1 = branch[i]
                    p2 = branch[i+1]
                    glBegin(GL_LINES)
                    glVertex3f(p1[0], p1[1], p1[2])
                    glVertex3f(p2[0], p2[1], p2[2])
                    glEnd()
            c += 1


    """
    Given: 2D point in viewing plane coordinates
    Returns: 3D point in global coordinates
    """
    def get3d_coord(self, point):
        a = point.x() * self.width_Scaled/800
        b = point.y() * self.height_Scaled/800
        coords = self.p + a * self.v1 + b * self.v2
        x = coords[0] / self.volumeWidth
        y = coords[1] / self.volumeWidth
        z = coords[2] / self.volumeWidth + (1 / 2 - (self.volumeLength / (self.volumeWidth * 2)))
        return [x,y,z]

    """
    Given:
        annotation: coordinates of annotation, either defining the vertices of a polygon or the coordinates of a junction point
        center: center of annotation
        type: either string "junction" or "polygon" indicating type of annotation. 
            Important for function renderAnnotations, which will render based on type of annotation
    Updates self.skeleton and self.annotations with new annotation
    """
    def add_annotation(self,annotation,center,type):

        if self.currBranch == 0:
            self.skeleton[self.currBranch].insert(self.currPolygon,center)

        else:
            self.skeleton[self.currBranch].insert(self.currPolygon+1,center)

        self.annotations[self.currBranch].insert(self.currPolygon,[type,annotation])

    """
    Called when the widget is updated in order to render the scene
    """
    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  # create a clear slate each time
        gl.glPushMatrix()  # push the current matrix to the current stack

        # describes the modeling matrix - viewing transformation. Condense into one matrix
        # get the inverse camera transformation matrix and apply to vector [0,0,0,1] to get the lighting
        # projection * viewing * modeling
        gluLookAt(self.pan[0], self.pan[1], 3.0, self.pan[0], self.pan[1], 0.0, 0.0, 1.0, 0.0)
        gl.glScale(self.scale, self.scale, self.scale)
        gl.glRotate(self.rotX, 1.0, 0.0, 0.0)
        gl.glRotate(self.rotY, 0.0, 1.0, 0.0)
        gl.glRotate(self.rotZ, 0.0, 0.0, 1.0)
        gl.glTranslate(-0.5, -0.5, -0.5)  # first, translate cube center to origin

        # if button skeleton is not clicked, then render annotations and text
        if self.renderAll:
            self.renderPlane()
            self.renderAnnotations()

        # render skeleton
        self.renderSkeleton()

        # if button ground truth is clicked, then render ground truth by calling the display list
        if self.showGT:
            glCallList(self.displayList)

        # render arrows
        self.renderArrow(1 * self.v1, np.array([1.0, 0.0, 0.0]))
        self.renderArrow(-1 * self.v2, np.array([0.0, 1.0, 0.0]))
        self.renderArrow(-1 * self.n, np.array([0.0, 0.0, 1.0]))

        # render - connect to the VBOs
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(2.0)
        gl.glColor3f(1.0, 1.0, 1.0)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.vertVBO)

        # render the cube faces of our image volume
        gl.glDrawElements(gl.GL_QUADS, len(self.cubeIdxArray), gl.GL_UNSIGNED_INT, self.cubeIdxArray)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glPopMatrix()
        glFlush()

    """
    Define image volume rectangular prism
    See coordinate system notes at the top for more information about OpenGL coordinate system
    """
    def initGeometry(self):
        # center is at (0.5,0.5,0.5)
        w = 1
        h = 1
        l = self.volumeLength/self.volumeWidth #transform length from global coordinates the OpenGL coordinates
        self.cubeVtxArray = np.array(

            [[0.5 - w / 2, 0.5 - h / 2, 0.5 - l / 2],
             [0.5 + w / 2, 0.5 - h / 2, 0.5 - l / 2],
             [0.5 + w / 2, 0.5 + h / 2, 0.5 - l / 2],
             [0.5 - w / 2, 0.5 + h / 2, 0.5 - l / 2],
             [0.5 - w / 2, 0.5 - h / 2, 0.5 + l / 2],
             [0.5 + w / 2, 0.5 - h / 2, 0.5 + l / 2],
             [0.5 + w / 2, 0.5 + h / 2, 0.5 + l / 2],
             [0.5 - w / 2, 0.5 + h / 2, 0.5 + l / 2]])

        # cast to floats and reshape
        self.vertVBO = vbo.VBO(np.reshape(self.cubeVtxArray,
                                          (1, -1)).astype(np.float32))
        # bind the VBO for use by GPU
        self.vertVBO.bind()

        # specify the vertices that make up a face
        self.cubeIdxArray = np.array(
            [0, 1, 2, 3,
             3, 2, 6, 7,
             1, 0, 4, 5,
             2, 1, 5, 6,
             0, 3, 7, 4,
             7, 6, 5, 4])

"""
Window on the left in the tool that displays the CT image scans and where the user draws annotations
This window is referred to as the "2D window" 
"""



        
class GraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
    
        
        super().__init__(parent)
        
        
        # width of window
        self.windowWidth = 800
        self.windowHeight = 800
        # width of viewing plane after scaled
        self.width_Scaled = 800
        self.height_Scaled = 800
        # scene object
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        # vectors and variables describing viewing plane
        self.v1 = np.array([1, 0, 0])
        self.v2 = np.array([0, 1, 0])
        self.n = np.cross(self.v1, self.v2)
        self.c = np.array([200, 200, 0])
        # variables for rotating and translating in 3d window
        self.startX = 0
        self.startY = 0
        self.startV1 = np.array([0, 0, 0])
        self.startV2 = np.array([0, 0, 0])
        self.startN = np.array([0, 0, 0])
        self.startC = np.array([0, 0, 0])
        # PyQT QGraphics objects and variables
        self.pixmap = QtWidgets.QGraphicsPixmapItem() # image displaying data set
        self.scene.addItem(self.pixmap)
        self.polygon = QtWidgets.QGraphicsPolygonItem(self.pixmap) # current annotated polygon
        self.junction = []  # current junction point annotated
        self.point = [] #junction point object
        self.tracking = []
        self.hierarchy = []
        self.junction_color = QtCore.Qt.red
        # booleans and variables determining settings for navigation
        self.shift_on = False
        self.alt_on = False
        self.F_on = False
        self.dontMove = False
        self.orient = False
        self.v1_rot = False
        self.v2_rot = False
        self.exp = 0
        self.wheel_denom = 120
        # booleans determining what is rendered in 2d window
        self.drawPolygon_Bool = False
        self.drawJunction_Bool = False
        self.drawTracking_Bool = False
        self.default_Bool = False
        self.defaultTracking_Bool = False
        self.center_Bool = False
        self.new_Default = False
        self.center_point = []
        self.center_struc = []
        self.center_color = QtCore.Qt.cyan
        self.glWidget = None
        self.def_poly = self.polygon.polygon()
        self.defaultType = "tracking"
        # see data structure notes above
        self.polygons_planes = []

    """
    Rotate about the normal vector (blue vector in OpenGL window)
    """
    def rot_about_N(self, theta, v1, v2, n):
        vec1 = v1 * math.cos(theta) + np.cross(n, v1) * math.sin(theta) + n * np.dot(n, v1) * (1 - math.cos(theta))
        vec2 = v2 * math.cos(theta) + np.cross(n, v2) * math.sin(theta) + n * np.dot(n, v2) * (1 - math.cos(theta))
        vecn = np.cross(vec1, vec2)
        return np.array([vec1, vec2, vecn])

    """
    Rotate about the vector v1 (red vector in OpenGL window)
    """
    def rot_about_V1(self, theta, v1, v2, n):
        vec2 = v2 * math.cos(theta) + np.cross(v1, v2) * math.sin(theta) + v1 * np.dot(v1, v2) * (1 - math.cos(theta))
        vecn = n * math.cos(theta) + np.cross(v1, n) * math.sin(theta) + v1 * np.dot(v1, n) * (1 - math.cos(theta))
        vec1 = np.cross(vec2, vecn)
        return np.array([vec1, vec2, vecn])

    """
    Rotate about the vector v1 (green vector in OpenGL window)
    """
    def rot_about_V2(self, theta, v1, v2, n):
        vec1 = v1 * math.cos(theta) + np.cross(v2, v1) * math.sin(theta) + v2 * np.dot(v2, v1) * (1 - math.cos(theta))
        vecn = n * math.cos(theta) + np.cross(v2, n) * math.sin(theta) + v2 * np.dot(v2, n) * (1 - math.cos(theta))
        vec2 = np.cross(vecn, vec1)
        return np.array([vec1, vec2, vecn])

    """
    When user scrolls with mouse in the 2D window, the user the viewing plane in the opposite direction of the normal vector 
    Self.c is updated to indicate where the viewing plane has moved. 
    """
    def wheelEvent(self, event):
        if (self.inBounds(event.x(), event.y())):
            numSteps = event.angleDelta().y() / self.wheel_denom
            self.c = np.subtract(self.c, self.n * numSteps)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Shift:
            self.dontMove = True
            self.shift_on = True
            
        if e.key() == Qt.Key_Alt:
            self.dontMove = True
            self.alt_on = True
        if e.key() == Qt.Key_F:
            plane = [self.startV1, self.startV2, self.startN, self.startC]
            if len(self.def_poly)>0:
                center = self.getCenter(self.def_poly)
                self.glWidget.add_annotation(self.def_poly,center,"polygon")
                self.resetPolygon()
                self.polygons_planes[self.glWidget.currBranch].insert(self.glWidget.currPolygon, plane)
                self.resetPolygon()
            self.dontMove = True
            self.F_on = True

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Shift:
            self.shift_on = False
            self.dontMove = False
        if e.key() == Qt.Key_Alt:
            self.alt_on = False
            self.dontMove = False
        if e.key() == Qt.Key_F:
            self.F_on = False
            self.dontMove = False
            

    """
    Given:
        poly: list of 2d window coordinates defining a polygon
    Returns coordinates converted to 3d global coordinates
    """
    def make_polygons(self,poly):
        lst_points = poly.polygon()
        poly = []
        for i in range(len(lst_points)):
            point = lst_points[i]
            coords = self.glWidget.get3d_coord(point)
            poly.append(coords)
        return poly

    """
    Given:
        poly: list of 3d global coordinates defining a polygon
    Return center of the polygon
    """
    def getCenter(self,poly):
        xSum = 0
        ySum = 0
        zSum = 0

        total = len(poly)
        for point in poly:
            xSum += point[0]
            ySum += point[1]
            zSum += point[2]
        return [xSum/total, ySum/total, zSum/total]

    """
    Resets current polygon
    """
    def resetPolygon(self):
        self.scene.removeItem(self.polygon)
        self.def_poly = self.polygon.polygon()
        self.polygon = QtWidgets.QGraphicsPolygonItem(self.pixmap)
        self.polygon.setPen(QtGui.QPen(QtCore.Qt.red, 3, QtCore.Qt.SolidLine))
        self.polygon.setBrush(QtGui.QBrush(QtCore.Qt.red))


    """
    When user presses the mouse in 2d window. 
    Used to draw polygons, draw junction and tracking points, and translate the viewing plane
    """
    def mousePressEvent(self, event):
        
        if self.drawPolygon_Bool:
            coord = self.pixmap.mapFromScene(self.mapToScene(event.pos()))
            poly = self.polygon.polygon()
            poly.append(coord)
            self.polygon.setPolygon(poly)
        elif self.drawJunction_Bool:
            coord = self.pixmap.mapFromScene(self.mapToScene(event.pos()))
            pen = QtGui.QPen(self.junction_color, 3, QtCore.Qt.SolidLine)
            brush = QtGui.QBrush(self.junction_color)
            point = self.scene.addRect(coord.x()-2, coord.y() - 2, 4, 4, pen, brush)
            self.junction = [point,coord]
        elif self.drawTracking_Bool:
            coord = self.pixmap.mapFromScene(self.mapToScene(event.pos()))
            pen = QtGui.QPen(self.junction_color, 3, QtCore.Qt.SolidLine)
            brush = QtGui.QBrush(self.junction_color)
            point = self.scene.addRect(coord.x()-2, coord.y() - 2, 4, 4, pen, brush)
            self.tracking = [point,coord]
        else:
            if (self.inBounds(event.x(), event.y())):
                if event.button() == Qt.LeftButton:
                    self.startX = event.x()
                    self.startY = event.y()
                    self.startV1 = self.v1
                    self.startV2 = self.v2
                    self.startN = self.n
                    self.startC = self.c
                    if self.shift_on == True and self.default_Bool == True:
                        
                        coord = self.pixmap.mapFromScene(self.mapToScene(event.pos()))
                        pen = QtGui.QPen(self.junction_color, 3, QtCore.Qt.SolidLine)
                        brush = QtGui.QBrush(self.junction_color)
                        point = self.scene.addRect(coord.x()-2, coord.y() - 2, 4, 4, pen, brush)
                        self.tracking = [point,coord]
                        
                        plane = [self.startV1, self.startV2, self.startN, self.startC]
                        if self.new_Default:
                            #self.glWidget.currBranch = len(self.glWidget.annotations)-1
                            self.glWidget.currPolygon = len(self.glWidget.annotations[self.glWidget.currBranch])+1
                        self.junction_color = self.glWidget.polygon_colors[self.glWidget.currBranch]
                        self.scene.removeItem(self.tracking[0])
                        if self.defaultType == "tracking":
                            self.tracking = [point,coord]
                            self.type = "tracking"
                            tracking = self.glWidget.get3d_coord(self.tracking[1])
                            self.glWidget.add_annotation(tracking, tracking, "tracking")
                            self.polygons_planes[self.glWidget.currBranch].insert(self.glWidget.currPolygon, plane)
                            self.tracking = []
                        elif self.defaultType == "junction":
                            self.junction = [point,coord]
                            self.type = "junction"
                            junction = self.glWidget.get3d_coord(self.junction[1])
                            self.glWidget.add_annotation(junction, junction, "junction")
                            self.polygons_planes[self.glWidget.currBranch].insert(self.glWidget.currPolygon, plane)
                            self.junction = []
                        else:
                            self.polygon.setPen(QtGui.QPen(self.junction_color, 3, QtCore.Qt.SolidLine))
                            self.polygon.setBrush(QtGui.QBrush(self.junction_color))
                            self.def_poly = self.polygon.polygon()
                            self.def_poly.append(coord)
                            self.polygon.setPolygon(self.def_poly)
                            self.def_poly = self.make_polygons(self.polygon)
                        
                        
                    if self.alt_on == True and self.default_Bool == True:
                        if self.new_Default:
                            #self.glWidget.currBranch = self.glWidget.currBranch-1
                            self.glWidget.currPolygon = len(self.glWidget.annotations[self.glWidget.currBranch])-1
                        annotation = self.glWidget.annotations[self.glWidget.currBranch][self.glWidget.currPolygon]
                        del self.glWidget.annotations[self.glWidget.currBranch][self.glWidget.currPolygon]
                        if self.glWidget.currBranch == 0:
                            del self.glWidget.skeleton[self.glWidget.currBranch][self.glWidget.currPolygon]
                        else:
                            del self.glWidget.skeleton[self.glWidget.currBranch][self.glWidget.currPolygon+1]
                        del self.polygons_planes[self.glWidget.currBranch][self.glWidget.currPolygon]

                        if annotation[0] == "junction":
                            for childBranch in self.hierarchy[self.glWidget.currBranch]:
                                if self.glWidget.skeleton[childBranch][0] == annotation[1]:
                                    del self.glWidget.skeleton[childBranch][0]
                        if annotation[0] == "tracking":
                            for childBranch in self.hierarchy[self.glWidget.currBranch]:
                                if self.glWidget.skeleton[childBranch][0] == annotation[1]:
                                    del self.glWidget.skeleton[childBranch][0]
                        
                        
                    
                    
                        

                if event.button() == Qt.RightButton:
                    self.startX = event.x()
                    self.startY = event.y()
                    self.startV1 = self.v1
                    self.startV2 = self.v2
                    self.startN = self.n
                    self.startC = self.c

                    # center at clicked point
                    if self.shift_on == True:
                        coord = self.pixmap.mapFromScene(self.mapToScene(event.pos()))
                        x = coord.x()
                        y = coord.y()
                        localCoord = np.array([x, y, 0])
                        # find where clicked coord is in global coord
                        u = (localCoord[0] - self.windowWidth / 2) * ((5 / 6) ** self.exp)
                        v = (localCoord[1] - self.windowWidth / 2) * ((5 / 6) ** self.exp)
                        globalCoord = self.c + u * self.v1 + v * self.v2
                        # move c and p
                        self.c = globalCoord
                        
                        
                        
                    
                        
                        
                       
                        

    """
    Determines is point is in range of 2d window
    """
    def inBounds(self, x, y):
        if x > 12 and x < 812 and y > 50 and y < 850:
            return True
        else:
            return False

    """
    When user moves the mouse in 2D window. 
    Used to rotate the viewing plane
    """
    def mouseMoveEvent(self, event):
        if self.drawPolygon_Bool == False:
            if self.drawJunction_Bool == False and self.drawTracking_Bool == False:
                if (self.inBounds(event.x(), event.y())):
                    x_disp = self.startX - event.x()
                    y_disp = self.startY - event.y()
                    if event.buttons() & Qt.LeftButton:
                        if self.shift_on == False:
                            if self.v1_rot ==False and self.v2_rot==False:
                                pass
                            else:
                                if self.v1_rot and self.v2_rot:
                                    v2 = self.rot_about_V2((x_disp * 0.005), self.startV1, self.startV2, self.startN)
                                    v1 = self.rot_about_V1((-y_disp * 0.005), self.startV1, self.startV2, self.startN)
                                    vn = np.array([0, 0, 0])
                                elif self.v1_rot:
                                    v1 = self.rot_about_V1((-y_disp * 0.005), self.startV1, self.startV2, self.startN)
                                    v2 = np.array([0, 0, 0])
                                    vn = np.array([0, 0, 0])
                                elif self.v2_rot:
                                    v2 = self.rot_about_V2((x_disp * 0.005), self.startV1, self.startV2, self.startN)
                                    v1 = np.array([0, 0, 0])
                                    vn = np.array([0, 0, 0])
                                else:
                                    v1 = np.array([0, 0, 0])
                                    v2 = np.array([0, 0, 0])
                                    vn = np.array([0, 0, 0])
                                self.v1 = (v1[0] + v2[0] + vn[0]) / np.linalg.norm(v1[0] + v2[0] + vn[0])
                                self.v2 = (v1[1] + v2[1] + vn[1]) / np.linalg.norm(v1[1] + v2[1] + vn[1])
                                self.n = (v1[2] + v2[2] + vn[2]) / np.linalg.norm(v1[2] + v2[2] + vn[2])
                        if self.shift_on == True:
                            v1 = np.array([0, 0, 0])
                            v2 = np.array([0, 0, 0])
                            vn = self.rot_about_N(y_disp * 0.005, self.startV1, self.startV2, self.startN)
                            self.v1 = (v1[0] + v2[0] + vn[0]) / np.linalg.norm(v1[0] + v2[0] + vn[0])
                            self.v2 = (v1[1] + v2[1] + vn[1]) / np.linalg.norm(v1[1] + v2[1] + vn[1])
                            self.n = (v1[2] + v2[2] + vn[2]) / np.linalg.norm(v1[2] + v2[2] + vn[2])
                    if event.buttons() & Qt.RightButton:
                        if self.dontMove == False:
                            self.c = self.startC + self.v1 * x_disp * ((5 / 6) ** self.exp) + self.v2 * y_disp * (
                                    (5 / 6) ** self.exp)

    """
    Given:
        intersections: a list of points where the viewing plane intersects with skeleton branches
    Draws points of intersection in the color of the branch of intersection
    """
    def drawIntersections(self,intersections):
        #remove points already painted
        if len(self.point) > 0:
            for point in self.point:
                self.scene.removeItem(point)

        #draw new points
        if len(intersections)>0:
            for point in intersections:
                x = point[0]
                y = point[1]
                color = point[2]
                pen = QtGui.QPen(color, 3, QtCore.Qt.SolidLine)
                brush = QtGui.QBrush(color)
                point = self.scene.addRect(x-2,y-2,4,4,pen,brush)
                self.point.append(point)
"""
PyQT Widgets
Creates interpolated image for 2d window
"""
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.volumeWidth = 400
        self.volumeHeight = 400
        self.volumeLength = 250
        self.windowWidth = 800
        self.windowHeight = 800
        self.v1 = np.array([1, 0, 0])
        self.v2 = np.array([0, 1,0])
        self.n = np.cross(self.v1, self.v2)
        self.c = np.array([self.volumeWidth / 2, self.volumeHeight / 2, self.volumeLength / 2])
        self.startX = 0
        self.startY = 0
        self.startV1 = np.array([0, 0, 0])
        self.startV2 = np.array([0, 0, 0])
        self.startN = np.array([0, 0, 0])
        self.startC = np.array([0, 0, 0])
        self.shift_on = False
        self.alt_on = False
        self.width_Scaled = self.windowWidth
        self.height_Scaled = self.windowHeight
        self.exp = 0
        self.values = []
        self.valuesGT = []
        self.points = []
        self.junctions = []
        self.trackings = []
        self.branch_Index = 0
        self._folder_path = ""
        self.wheel_denom = 120
        self.currX = 0
        self.currY = 0
        self.dontMove=False
        self.showGT = False
        self.showGT_img = False
        self.loadImgs = False
        self.loadGT = False
        self.glWidget = GLWidget(self)
        
        self.initUI()
        self.initGUI()

        # call paintGL with updateGL which calls paintGL automatically. Rendering happens periodically
        # call showPlane. Updates 2d window with correct interpolated pixmap image periodically
        timer = QtCore.QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.glWidget.updateGL)
        timer.timeout.connect(self.showPlane)

        timer.start()

    """
    PyQT Menu bar
    """
    def initUI(self):
        save = QAction('Save As...', self)
        save.setShortcut('Ctrl+S')
        save.setStatusTip('Open new File')
        save.triggered.connect(self.save)

        open = QAction('Open...', self)
        open.setShortcut('Ctrl+O')
        open.setStatusTip('Open saved work')
        open.triggered.connect(self.open)

        openFile = QAction('Load Files', self)
        openFile.setShortcut('Ctrl+L')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.loadImages)

        openGroundTruth = QAction('Load Ground Truth', self)
        openGroundTruth.setShortcut('Ctrl+G')
        openGroundTruth.setStatusTip('Open new File')
        openGroundTruth.triggered.connect(self.loadGroundTruth)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(save)
        fileMenu.addAction(open)
        fileMenu.addAction(openFile)
        fileMenu.addAction(openGroundTruth)

        self.view = GraphicsView()
        self.view.glWidget = self.glWidget
        self.setWindowTitle("Root Segmentation")
        self.setGeometry(300, 300, 2000, 900)
        self.show()

    """
    PyQT layout and buttons
    """
    def initGUI(self):
        central_widget = QWidget()
        gui_layout = QVBoxLayout()
        toolbar1 = QHBoxLayout()
        sidebar = QVBoxLayout()
        toolbar2 = QHBoxLayout()
        toolbar3 = QHBoxLayout()
        toolbar4 = QHBoxLayout()

        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        vbox3 = QVBoxLayout()
        hbox = QHBoxLayout()
        threeD = QHBoxLayout()
        vbox1.addLayout(toolbar1)
        vbox1.addLayout(gui_layout)
        vbox2.addLayout(toolbar2)
        vbox2.addLayout(threeD)
        hbox.addLayout(sidebar)
        
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)
        central_widget.setLayout(hbox)
        self.setCentralWidget(central_widget)
        

        branch_button = QPushButton(central_widget)
        branch_button.setText("Create Branch")
        branch_button.clicked.connect(self.branch_button_clicked)

        goTo_button = QPushButton(central_widget)
        goTo_button.setText("Go To Annotation")
        goTo_button.clicked.connect(self.goTo_button_clicked)

        removeBranch_button = QPushButton(central_widget)
        removeBranch_button.setText("Remove Branch")
        removeBranch_button.clicked.connect(self.removeBranch_button_clicked)

        remove_button = QPushButton(central_widget)
        remove_button.setText("Remove Annotation")
        remove_button.clicked.connect(self.remove_button_clicked)

        create_button = QPushButton(central_widget)
        create_button.setText("Annotate")
        create_button.clicked.connect(self.create_button_clicked)

        clear_button = QPushButton(central_widget)
        clear_button.setText("Clear Annotation")
        clear_button.clicked.connect(self.clear_button_clicked)

        finish_button = QPushButton(central_widget)
        finish_button.setText("Finish Annotation")
        finish_button.clicked.connect(self.finish_button_clicked)

        top_button = QPushButton(central_widget)
        top_button.setText("Top")
        top_button.clicked.connect(self.top_button_clicked)

        bottom_button = QPushButton(central_widget)
        bottom_button.setText("Bottom")
        bottom_button.clicked.connect(self.bottom_button_clicked)

        left_button = QPushButton(central_widget)
        left_button.setText("Left")
        left_button.clicked.connect(self.left_button_clicked)

        right_button = QPushButton(central_widget)
        right_button.setText("Right")
        right_button.clicked.connect(self.right_button_clicked)

        back_button = QPushButton(central_widget)
        back_button.setText("Back")
        back_button.clicked.connect(self.back_button_clicked)

        front_button = QPushButton(central_widget)
        front_button.setText("Front")
        front_button.clicked.connect(self.front_button_clicked)

        XY_button = QPushButton(central_widget)
        XY_button.setText("XY")
        XY_button.clicked.connect(self.XY_button_clicked)

        XZ_button = QPushButton(central_widget)
        XZ_button.setText("XZ")
        XZ_button.clicked.connect(self.XZ_button_clicked)

        YZ_button = QPushButton(central_widget)
        YZ_button.setText("YZ")
        YZ_button.clicked.connect(self.YZ_button_clicked)

        self.v1_rot_checkbox = QCheckBox("Top/Down Rotate")
        self.v1_rot_checkbox.stateChanged.connect(self.v1_rot_toggle)
        self.v1_rot_checkbox.setChecked(True)

        self.v2_rot_checkbox = QCheckBox("Left/Right Rotate")
        self.v2_rot_checkbox.stateChanged.connect(self.v2_rot_toggle)

        self.gt_checkbox = QCheckBox("ground truth")
        self.gt_checkbox.stateChanged.connect(self.groundTruth_toggle)

        self.skeleton_checkbox = QCheckBox("skeleton")
        self.skeleton_checkbox.stateChanged.connect(self.skeleton_toggle)
        
        self.center_checkbox = QCheckBox("center")
        self.center_checkbox.stateChanged.connect(self.center_toggle)
        
        self.default_checkbox = QCheckBox("default")
        self.default_checkbox.stateChanged.connect(self.default_toggle)
        
        perp_button = QPushButton(central_widget)
        perp_button.setText("Ortho")
        perp_button.clicked.connect(self.perp_button_clicked)
        
        self.branch_combobox = QComboBox(self)
        self.branch_combobox.activated.connect(self.updateIndexCombo)
        self.branch_combobox.activated.connect(self.updateIndexCombo)
        
        self.index_combobox = QComboBox(self)
        self.index_combobox.currentIndexChanged.connect(self.updateCurrentParams)
        
        self.type_combobox = QComboBox(self)
        self.type_combobox.addItems(["tracking", "junction", "polygon"])
        self.type_combobox.activated.connect(self.updateTypeDefault)
        #self.updateIndexCombo(self.branch_combobox.currentIndex())
        
        

        scaleText = QLabel(self)
        scaleText.setText("Scale Plane")
        scaleText.setAlignment(Qt.AlignBottom | Qt.AlignRight)

        wheelText = QLabel(self)
        wheelText.setText("Step Size")
        wheelText.setAlignment(Qt.AlignBottom | Qt.AlignRight)

        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(0)
        self.scale_slider.setMaximum(20)
        self.scale_slider.setValue(0)
        self.scale_slider.valueChanged.connect(self.ratio_change)

        self.wheel_slider = QSlider(Qt.Horizontal)
        self.wheel_slider.setMinimum(1)
        self.wheel_slider.setMaximum(15)
        self.wheel_slider.setValue(1)
        self.wheel_slider.valueChanged.connect(self.wheel_change)

        sidebar.addWidget(self.v1_rot_checkbox)
        sidebar.addWidget(self.v2_rot_checkbox)
        sidebar.addLayout(toolbar4)
        toolbar4.addWidget(self.default_checkbox)
        toolbar4.addWidget(perp_button)
        sidebar.addLayout(toolbar3)
        toolbar3.addWidget(self.branch_combobox)
        toolbar3.addWidget(self.index_combobox)
        sidebar.addWidget(self.type_combobox)
        sidebar.addWidget(goTo_button)
        sidebar.addWidget(removeBranch_button)
        sidebar.addWidget(remove_button)
        sidebar.addWidget(branch_button)
        sidebar.addWidget(create_button)
        sidebar.addWidget(clear_button)
        sidebar.addWidget(finish_button)
        toolbar1.addWidget(top_button)
        toolbar1.addWidget(bottom_button)
        toolbar1.addWidget(left_button)
        toolbar1.addWidget(right_button)
        toolbar1.addWidget(back_button)
        toolbar1.addWidget(front_button)
        toolbar1.addWidget(XY_button)
        toolbar1.addWidget(XZ_button)
        toolbar1.addWidget(YZ_button)
        toolbar2.addWidget(scaleText)
        toolbar2.addWidget(self.scale_slider)
        toolbar2.addWidget(wheelText)
        toolbar2.addWidget(self.wheel_slider)
        toolbar2.addWidget(self.gt_checkbox)
        toolbar2.addWidget(self.skeleton_checkbox)
        toolbar2.addWidget(self.center_checkbox)
        

        # openGL and QLabel
        gui_layout.addWidget(self.view)
        threeD.addWidget(self.glWidget)
        
    """
    Determines the type requested from the combo box
    """
    def updateTypeDefault(self, index):
        if index == 0:
            self.view.defaultType = "tracking"
        elif index == 1:
            self.view.defaultType = "junction"
        else:
            self.view.defaultType = "polygon"
        
    """
    updates currBranch and currPolygon
    """
    def updateCurrentParams(self):
        if self.index_combobox.currentText() != "":
            self.glWidget.currBranch = int(self.branch_combobox.currentText()) - 1
            if self.index_combobox.currentText() == "new":
                self.view.new_Default = True
            elif self.branch_combobox.currentText() != "" and self.index_combobox.currentText() != "":
                self.view.new_Default = False
                self.glWidget.currPolygon = int(self.index_combobox.currentText()) - 1
    
    """
    Fills the polygon combo boxes
    """
    def updateIndexCombo(self,index):
        self.index_combobox.clear()
        self.branch_Index = index
        length = len(self.glWidget.annotations[self.branch_Index])
        polygons = self.getOptions(length)
        self.index_combobox.addItems(polygons)
        self.index_combobox.addItem("new")

    """
    Allows the continuous mode to be turned on and off
    """
    def default_toggle(self):
        if self.default_checkbox.isChecked():
            self.view.default_Bool = True
        else:
            self.view.default_Bool = False

    """
    When the center checkbox is clicked, a point showing the center of the 2D window will appear
    """
    def center_toggle(self):
        if self.center_checkbox.isChecked():
            if self.view.center_Bool == False:
                pen = QtGui.QPen(self.view.center_color, 3, QtCore.Qt.SolidLine)
                brush = QtGui.QBrush(self.view.center_color)
                self.view.center_point = self.view.scene.addRect(399, 399, 1, 1, pen, brush)
                self.view.center_struc = [self.view.center_point, [399,399]]
                self.view.center_Bool = True
        else:
            if len(self.view.center_struc)>0:
                self.view.scene.removeItem(self.view.center_struc[0])
                self.view.center_Bool = False
                        
            
    """
    When the skeleton checkbox is clicked, only the skeleton of the branches is rendered
    Otherwise, the annotations, text, and skeleton is annotated 
    """
    def skeleton_toggle(self):
        if self.skeleton_checkbox.isChecked() == True:
            self.glWidget.renderAll = False
        else:
            self.glWidget.renderAll = True

    """
    When Top/Down checkbox is clicked, the user only rotates with respect to v1
    """
    def v1_rot_toggle(self):
        if self.v1_rot_checkbox.isChecked() == True:
            self.view.v1_rot = True
        else:
            self.view.v1_rot = False

    """
    When Top/Down checkbox is clicked, the user only rotates with respect to v2
    """
    def v2_rot_toggle(self):
        if self.v2_rot_checkbox.isChecked() == True:
            self.view.v2_rot = True
        else:
            self.view.v2_rot = False

    """
    When ground truth checkbox is clicked, ground truth is rendered. 
    The user can only render the ground truth when they have loaded the ground truth 
    """
    def groundTruth_toggle(self):
        if self.gt_checkbox.isChecked() == True:
            self.glWidget.showGT = True
        else:
            self.glWidget.showGT = False
    

    """
    Creates a new branch
    """
    def branch_button_clicked(self):
        color = QColorDialog.getColor()
        if len(self.glWidget.annotations)>0: # if first branch, do not need a junction point
            branches = self.getOptions(len(self.glWidget.annotations))

            # get parent branch for junction point
            parentBranch, next = QInputDialog.getItem(self, 'Get Parent Branches',
                                                    'Select Parent Branch:', branches, self.glWidget.currBranch)
            if next:
                parentBranch = int(parentBranch)-1
                length = len(self.glWidget.annotations[self.glWidget.currBranch])
                junctions = self.getOptions(length)
                # get index for junction point
                num, ok = QInputDialog.getItem(self, 'Select Junction',
                                               'Select Index:', junctions)

                if ok:
                    # add junction point to front of skeleton, update data structures
                    self.glWidget.skeleton.append([self.glWidget.annotations[parentBranch][int(num)-1][1]])
                    createdBranch = len(self.glWidget.annotations)
                    self.view.hierarchy[parentBranch].append(createdBranch)
                    self.glWidget.currBranch += 1
                    self.view.hierarchy.append([])
                    self.view.polygons_planes.append([])
                    self.glWidget.annotations.append([])
                    self.glWidget.polygon_colors.append(color)
                    self.branch_combobox.addItem(str(len(branches)+1))
                    self.index_combobox.clear()
                    self.index_combobox.addItem("new")
        else:
            # update data structures
            self.glWidget.skeleton.append([])
            self.glWidget.currBranch += 1
            self.view.hierarchy.append([])
            self.view.polygons_planes.append([])
            self.glWidget.annotations.append([])
            self.glWidget.polygon_colors.append(color)
            self.branch_combobox.addItems(self.getOptions(len(self.glWidget.annotations)))
            self.index_combobox.clear()
            self.index_combobox.addItem("new")

    """
    Given:
        num: length of options
    Returns a list of string of numbers from 1 - length
    For example, if num=2, then list returned has strings "1" and "2" 
    
    This is used for buttons that ask for the user to select a number to indicate branch and/or index
    """

    def getOptions(self,num):
        lst = []
        for i in range(1,num+1):
            lst.append(str(i))
        return lst


    """
    Go to annotation
    """
    def goTo_button_clicked(self):
        branches = self.getOptions(len(self.glWidget.annotations))
        # ask user which annotation they want to go to
        currBranch, next = QInputDialog.getItem(self, 'Go To Branch',
                                                'Select Branch:', branches, self.glWidget.currBranch)
        if next:
            self.glWidget.currBranch = int(currBranch) - 1
            length = len(self.glWidget.annotations[self.glWidget.currBranch])
            polygons = self.getOptions(length)
            num, ok = QInputDialog.getItem(self, 'Go To Annotation',
                                           'Select Index:', polygons, length - 1)
            if ok:
                # get plane v1,v2, n, and c for annotation of interest, update variables
                plane = self.view.polygons_planes[self.glWidget.currBranch][int(num)-1]
                print("plane",plane)
                self.view.v1 = plane[0]
                self.view.v2 = plane[1]
                self.view.n = plane[2]
                self.view.c = plane[3]
                self.glWidget.v1 = plane[0]
                self.glWidget.v2 = plane[1]
                self.glWidget.n = plane[2]
                self.glWidget.c = plane[3]

    """
    Remove branch
    """
    def removeBranch_button_clicked(self):
        branches = self.getOptions(len(self.glWidget.annotations))
        # ask user which branch they want to remove
        currBranch, ok = QInputDialog.getItem(self, 'Get Branches',
                                                'Select Branch:', branches, self.glWidget.currBranch)
        if ok:
            # remove by updating all data structures
            currBranch = int(currBranch)-1
            del self.glWidget.annotations[currBranch]
            del self.glWidget.polygon_colors[currBranch]
            del self.glWidget.skeleton[currBranch]
            del self.view.polygons_planes[currBranch]
            self.branch_combobox.removeItem(len(branches)-1)
            for childBranch in self.view.hierarchy[currBranch]:
                del self.glWidget.skeleton[childBranch-1][0]

    """
    Annotation
    """
    def remove_button_clicked(self):
        branches = self.getOptions(len(self.glWidget.annotations))
        # ask user which annotation they want to remove
        currBranch, next = QInputDialog.getItem(self, 'Get Branches',
                                                'Select Branch:', branches, self.glWidget.currBranch)
        if next:
            self.glWidget.currBranch = int(currBranch) - 1
            length = len(self.glWidget.annotations[self.glWidget.currBranch])
            polygons = self.getOptions(length)
            num, ok = QInputDialog.getItem(self, 'Add Polygon',
                                           'Select Index:', polygons, length - 1)

            if ok:
                # remove by updating all data structures
                currBranch = int(currBranch)-1
                num = int(num) -1
                annotation = self.glWidget.annotations[currBranch][num]
                del self.glWidget.annotations[currBranch][num]
                if currBranch == 0:
                    del self.glWidget.skeleton[currBranch][num]
                else:
                    del self.glWidget.skeleton[currBranch][num+1]
                del self.view.polygons_planes[currBranch][num]

                if annotation[0] == "junction":
                    for childBranch in self.view.hierarchy[currBranch]:
                        if self.glWidget.skeleton[childBranch][0] == annotation[1]:
                            del self.glWidget.skeleton[childBranch][0]
                if annotation[0] == "tracking":
                    for childBranch in self.view.hierarchy[currBranch]:
                        if self.glWidget.skeleton[childBranch][0] == annotation[1]:
                            del self.glWidget.skeleton[childBranch][0]
        

       
    """
    Create new annotation
    """

    
    
    def create_button_clicked(self):
        branches = self.getOptions(len(self.glWidget.annotations))
        currBranch, next = QInputDialog.getItem(self, 'Get Branches',
                                                'Select Branch:', branches, self.glWidget.currBranch)
        if next:
            self.glWidget.currBranch = int(currBranch)-1
            length = len(self.glWidget.annotations[self.glWidget.currBranch])+1
            polygons = self.getOptions(length)
            num, ok = QInputDialog.getItem(self, 'Add Polygon',
                                           'Select Index:', polygons, length-1)

            if ok:
                self.glWidget.currPolygon = int(num)-1
                type, final = QInputDialog.getItem(self, 'Type of Annotation',
                                                        'Select Annotation:', ["polygon", "junction point", "tracking point"])

                if final:
                    if type == "polygon":
                        color = self.glWidget.polygon_colors[self.glWidget.currBranch]
                        self.view.polygon.setPen(QtGui.QPen(color, 3, QtCore.Qt.SolidLine))
                        self.view.polygon.setBrush(QtGui.QBrush(color))
                        self.view.drawPolygon_Bool = True
                        self.type = "polygon"

                    else:
                        if type == "junction point":
                            self.view.junction_color = self.glWidget.polygon_colors[self.glWidget.currBranch]
                            self.view.drawJunction_Bool = True
                            self.type = "junction"
                        if type == "tracking point":
                            self.view.junction_color = self.glWidget.polygon_colors[self.glWidget.currBranch]
                            self.view.drawTracking_Bool = True
                            self.type = "tracking"
                        

    """
    Erase current annotation
    """
    def clear_button_clicked(self):
        self.view.drawPolygon_Bool = False
        self.view.drawJunction_Bool = False
        self.view.drawTracking_Bool = False
        self.view.resetPolygon()
        if len(self.view.junction)>0:
            self.view.scene.removeItem(self.view.junction[0])
            self.view.junction = []
        if len(self.view.tracking)>0:
            self.view.scene.removeItem(self.view.tracking[0])
            self.view.tracking = []

    
    """
    When the user finishes creating their annotation, updates data structures and renders annotation in 3D window
    """
    def finish_button_clicked(self):
        plane = [self.view.v1, self.view.v2, self.view.n, self.view.c]
        if self.view.defaultTracking_Bool:
            self.glWidget.currBranch = len(self.glWidget.annotations)-1
            self.glWidget.currPolygon = len(self.glWidget.annotations[self.glWidget.currBranch])+1
            self.view.junction_color = self.glWidget.polygon_colors[self.glWidget.currBranch]
                        
            self.type = "tracking"
            if len(self.view.tracking)>0:
                self.view.drawTracking_Bool = False
                self.view.scene.removeItem(self.view.tracking[0])
                tracking = self.glWidget.get3d_coord(self.view.tracking[1])
                self.glWidget.add_annotation(tracking, tracking, "tracking")
                self.view.polygons_planes[self.glWidget.currBranch].insert(self.glWidget.currPolygon, plane)
                self.view.tracking = []
            
        if self.type == "polygon":
            self.view.drawPolygon_Bool = False
            poly = self.view.make_polygons(self.view.polygon)
            if len(poly)>0:
                center = self.view.getCenter(poly)
                self.glWidget.add_annotation(poly,center,"polygon")
                self.view.resetPolygon()
                self.view.polygons_planes[self.glWidget.currBranch].insert(self.glWidget.currPolygon, plane)

        if self.type == "junction":
            if len(self.view.junction)>0:
                self.view.drawJunction_Bool = False
                self.view.scene.removeItem(self.view.junction[0])
                junction = self.glWidget.get3d_coord(self.view.junction[1])
                self.glWidget.add_annotation(junction, junction, "junction")
                self.view.polygons_planes[self.glWidget.currBranch].insert(self.glWidget.currPolygon, plane)
                self.view.junction = []
        if self.type == "tracking":
            if len(self.view.tracking)>0:
                self.view.drawTracking_Bool = False
                self.view.scene.removeItem(self.view.tracking[0])
                tracking = self.glWidget.get3d_coord(self.view.tracking[1])
                self.glWidget.add_annotation(tracking, tracking, "tracking")
                self.view.polygons_planes[self.glWidget.currBranch].insert(self.glWidget.currPolygon, plane)
                self.view.tracking = []


    """
    Sets the view orthogonal to the current line
    """
    def perp_button_clicked(self):
        if self.branch_combobox.currentText() != "":
            currBranch = int(self.branch_combobox.currentText())-1
            length = len(self.glWidget.annotations[currBranch])-1
            if length >= 1:
                if self.index_combobox.currentText() == "new" or self.index_combobox.currentText() == "1":
                    currCoord = self.glWidget.annotations[currBranch][length][1]
                    prevCoord = self.glWidget.annotations[currBranch][length-1][1]
                    #self.glWidget.c = self.view.polygons_planes[currBranch][length][3]
                else:
                    index = int(self.index_combobox.currentText())-1
                    currCoord = self.glWidget.annotations[currBranch][index][1]
                    prevCoord = self.glWidget.annotations[currBranch][index-1][1]
                    #self.glWidget.c = self.view.polygons_planes[currBranch][index][3]
                depthVector = np.array(currCoord) - np.array(prevCoord)
                normDepthVector = depthVector / np.linalg.norm(depthVector)
                normNormalVector = self.view.n / np.linalg.norm(self.view.n)
                axis = np.cross(normNormalVector,normDepthVector)
                angle = math.acos(np.dot(normNormalVector, normDepthVector))
                self.view.n = depthVector / np.linalg.norm(depthVector)
                #self.view.c =  np.array([self.volumeWidth / 2, self.volumeHeight / 2, 0])
                newVectors = self.view.rot_about_N(angle, self.v1, self.v2, axis)
                self.view.v1 = newVectors[0]
                self.view.v2 = newVectors[1]
                center = np.array(currCoord)
                self.view.c = np.array([center[0]*400 , center[1]*400, center[2]*200])
                print(currCoord)
                print(self.view.c)
            

    """
    Sets viewing plane to top of image volume 
    """
    def top_button_clicked(self):
        self.view.v1 = np.array([1, 0, 0])
        self.view.v2 = np.array([0, 1, 0])
        self.view.n = np.cross(self.v1, self.v2)
        self.view.c = np.array([self.volumeWidth / 2, self.volumeHeight / 2, 0])

    """
    Sets viewing plane to bottom of image volume 
    """
    def bottom_button_clicked(self):
        self.view.v1 = np.array([1, 0, 0])
        self.view.v2 = np.array([0, 1, 0])
        self.view.n = np.cross(self.v1, self.v2)
        self.view.c = np.array([self.volumeWidth / 2, self.volumeHeight / 2, self.volumeLength - 3])

    """
    Sets viewing plane to left of image volume 
    """
    def left_button_clicked(self):
        self.view.v1 = np.array([0, 1, 0])
        self.view.v2 = np.array([0, 0, 1])
        self.view.n = np.cross(self.v1, self.v2)
        self.view.c = np.array([0, self.volumeHeight / 2, self.volumeLength / 2])

    """
    Sets viewing plane to right of image volume 
    """
    def right_button_clicked(self):
        self.view.v1 = np.array([0, 1, 0])
        self.view.v2 = np.array([0, 0, 1])
        self.view.n = np.cross(self.v1, self.v2)
        self.view.c = np.array([self.volumeWidth - 1, self.volumeHeight / 2, self.volumeLength / 2])

    """
    Sets viewing plane to back of image volume 
    """
    def back_button_clicked(self):
        self.view.v1 = np.array([1, 0, 0])
        self.view.v2 = np.array([0, 0, 1])
        self.view.n = np.cross(self.v1, self.v2)
        self.view.c = np.array([self.volumeWidth / 2, 0, self.volumeLength / 2])

    """
    Sets viewing plane to front of image volume 
    """
    def front_button_clicked(self):
        self.view.v1 = np.array([1, 0, 0])
        self.view.v2 = np.array([0, 0, 1])
        self.view.n = np.cross(self.v1, self.v2)
        self.view.c = np.array([self.volumeWidth / 2, self.volumeHeight - 1, self.volumeLength / 2])

    """
    Sets viewing plane to be parallel to XY plane
    """
    def XY_button_clicked(self):
        self.view.v1 = np.array([1, 0, 0])
        self.view.v2 = np.array([0, 1, 0])
        self.view.n = np.cross(self.view.v1, self.view.v2)

    """
    Sets viewing plane to be parallel to XZ plane
    """
    def XZ_button_clicked(self):
        self.view.v1 = np.array([1, 0, 0])
        self.view.v2 = np.array([0, 0, 1])
        self.view.n = np.cross(self.view.v1, self.view.v2)

    """
    Sets viewing plane to be parallel to YZ plane
    """
    def YZ_button_clicked(self):
        self.view.v1 = np.array([0, 1, 0])
        self.view.v2 = np.array([0, 0, 1])
        self.view.n = np.cross(self.view.v1, self.view.v2)

    """
    Scales viewing plane width and height based on the scale slider
    """
    def ratio_change(self):
        self.view.exp = self.scale_slider.value()
        self.exp = self.scale_slider.value()
        self.width_Scaled = int(self.windowWidth * (5 / 6) ** self.exp)
        self.height_Scaled = int(self.windowHeight * (5 / 6) ** self.exp)
        self.showPlane()

    """
    Determines how fast the viewer scrolls through image volume
    """
    def wheel_change(self):
        val = self.wheel_slider.value()
        self.view.wheel_denom = int(120 / val)


    """
    updates variables across classes
    """
    def setVariables(self, w, h, l):
        self.volumeWidth = w
        self.volumeHeight = h
        self.volumeLength = l
        self.glWidget.volumeWidth = w
        self.glWidget.volumeHeight = h
        self.glWidget.volumeLength = l
        self.c = np.array([w / 2, h / 2, 0])

    """
    Saves work by recording data structures to a text file
    """
    def save(self):
        dialog = QFileDialog()
        name = dialog.getSaveFileName(self, 'Save File')[0]
        if name != "":
            f = open(name, "w")
            branches = self.glWidget.annotations
            planes = self.view.polygons_planes
            f.write(str(len(branches))+"\n")
            b=0
            # for each branch, rgb colors and num polygons in that branch
            for branch in branches:
                for i in range (len(self.view.hierarchy[b])):
                    f.write(str(self.view.hierarchy[b][i]) + "\n")
                f.write("next"+"\n")
                color = self.glWidget.polygon_colors[b]
                f.write(str(color.red())+"\n")
                f.write(str(color.green())+"\n")
                f.write(str(color.blue())+"\n")
                f.write(str(len(branch))+"\n")

               # for each polygon in the branch
                for i in range (len(branch)):
                    # get plane for polygon
                    for j in range (4):
                        for k in range(3):
                            f.write(str(planes[b][i][j][k])+"\n")

                    annotation = branch[i]
                    polygon = annotation[1]
                    print("polygon",polygon)
                    if annotation[0] == "junction" or annotation[0] == "tracking":
                        f.write(str(0)+"\n")
                        f.write(str(polygon[0])+"\n")
                        f.write(str(polygon[1])+"\n")
                        f.write(str(polygon[2])+"\n")
                    else:
                        f.write(str(1) +"\n")
                        for coord in polygon:
                            # get coords for polygon
                            f.write(str(coord[0])+"\n")
                            f.write(str(coord[1])+"\n")
                            f.write(str(coord[2])+"\n")
                        f.write("next"+"\n") # indicate done with coords

                #get the junction point for every branch that is not the main branch
                if b > 0:
                    f.write(str(self.glWidget.skeleton[b][0][0])+"\n")
                    f.write(str(self.glWidget.skeleton[b][0][1])+"\n")
                    f.write(str(self.glWidget.skeleton[b][0][2])+"\n")
                b += 1
            f.close()

    """
    Opens saved work by loading data structures from text file
    """
    def open(self):
        dialog = QFileDialog()
        file = dialog.getOpenFileName(None, "Select File")[0]
        if file != "":
            f = open(file,"r")
            self.glWidget.annotations = []
            self.glWidget.skeleton = []
            self.glWidget.polygon_colors = []
            self.view.polygons_planes = []
            self.view.hierarchy = []
            self.branch_combobox.clear()
            self.index_combobox.clear()
            numBranches = int(f.readline())

        # for each branch, rgb colors and num polygons in that branch
            for i in range(numBranches):
                curr = f.readline()
                print("currStart",curr)
                branchChildren = []
                self.branch_combobox.addItem(str(i+1))
                while curr != "next\n":
                    branchChildren.append(int(curr))
                    curr = f.readline()
                    print("curr in loop", curr)
                self.view.hierarchy.append(branchChildren)
                print("curr", curr)
                r = int(f.readline())
                g = int(f.readline())
                b = int(f.readline())
                color = QtGui.QColor(r, g, b)
                self.glWidget.polygon_colors.append(color)
                numPolygons = int(f.readline())
                branch = []
                branchPlanes = []
                branchSkeleton = []
                for j in range(numPolygons): #for each polygon in the branch
                    annotation = []
                    plane = []
                    # get plane for polygon
                    v1 = []
                    v1.append(float(f.readline()))
                    v1.append(float(f.readline()))
                    v1.append(float(f.readline()))
                    v1 = np.array(v1)
                    v2 = []
                    v2.append(float(f.readline()))
                    v2.append(float(f.readline()))
                    v2.append(float(f.readline()))
                    v2 = np.array(v2)
                    vn = []
                    vn.append(float(f.readline()))
                    vn.append(float(f.readline()))
                    vn.append(float(f.readline()))
                    vn = np.array(vn)
                    c = []
                    c.append(float(f.readline()))
                    c.append(float(f.readline()))
                    c.append(float(f.readline()))
                    c = np.array(c)
                    plane.append(v1)
                    plane.append(v2)
                    plane.append(vn)
                    plane.append(c)
                    print("plane",plane)
                    curr = f.readline()
                    if float(curr) == 0:
                        print("in junction")
                        annotation.append("junction")
                        curr = f.readline()
                        coord = []
                        coord.append(float(curr))
                        curr = f.readline()
                        coord.append(float(curr))
                        curr = f.readline()
                        coord.append(float(curr))
                        annotation.append(coord)
                        branch.append(annotation)
                        branchSkeleton.append(coord)
                        branchPlanes.append(plane)
                        

                    else:
                        polygon = []
                        annotation.append("polygon")
                        curr = f.readline()
                        while curr != "next\n": # get coords for polygon
                            coord = []
                            coord.append(float(curr))
                            curr = f.readline()
                            coord.append(float(curr))
                            curr = f.readline()
                            coord.append(float(curr))
                            curr = f.readline()
                            polygon.append(coord)
                        annotation.append(polygon)
                        branch.append(annotation)
                        center = self.view.getCenter(polygon)
                        branchSkeleton.append(center)
                        branchPlanes.append(plane)

                self.glWidget.annotations.append(branch)
                self.view.polygons_planes.append(branchPlanes)
                junction = []
                # DONT ADD SKELETON IF IT IS THE FIRST BRANCH
                if i>0:
                    junction.append(float(f.readline()))
                    junction.append(float(f.readline()))
                    junction.append(float(f.readline()))
                    branchSkeleton.insert(0,junction)
                    self.glWidget.skeleton.append(branchSkeleton)
                else:
                    self.glWidget.skeleton.append(branchSkeleton)
                print("skeleton",self.glWidget.skeleton)
    """
    User selects directory indicated by user with CT images of data
    Loads selected data and saves as matrices using function self.getValues() 
    """
    def loadImages(self):
        # Open a File Dialog and select the folder path
        dialog = QFileDialog()
        self._folder_path = dialog.getExistingDirectory(None, "Select Folder")
        if self._folder_path != "":

            # Get the list of images in the folder and read using matplotlib
            self.list_of_images = os.listdir(self._folder_path)
            self.images = sorted(self.list_of_images)
            self.setVariables(401, 401, len(self.list_of_images))
            values = self.getValues()
            points = self.getPoints()
            self.interpolate = RegularGridInterpolator(points,values,method='linear', bounds_error=False, fill_value=0)
            self.glWidget.initGeometry()
            self.loadImgs = True
            self.showPlane()

    """
    User selects directory indicated by user with ground truth data
    Loads selected data and saves as matrices using function self.getValues() 
    """
    def loadGroundTruth(self):
        # Open a File Dialog and select the folder path
        dialog = QFileDialog()
        self._folder_path = dialog.getExistingDirectory(None, "Select Folder")
        if self._folder_path != "":

            # Get the list of images in the folder and read using matplotlib
            start_time = time.time()
            self.list_of_images = os.listdir(self._folder_path)
            self.imagesGT = sorted(self.list_of_images)
            self.setVariables(401, 401, len(self.list_of_images))
            valuesGT = self.getValuesGT()
            points = self.getPoints()
            self.interpolateGT = RegularGridInterpolator(points,np.transpose(valuesGT),method='linear', bounds_error=False, fill_value=0)
            self.glWidget.initGeometry()
            self.glWidget.values = valuesGT
            self.glWidget.renderGroundTruth()
            self.loadGT = True
            #print("--- %s seconds ---" % (time.time() - start_time))

    """
    Returns 3D matrix of points defining image volume
    Used for creating interpolated images
    """

    def getPoints(self):
        x = np.arange(0, self.volumeWidth)
        y = np.arange(0, self.volumeHeight)
        z = np.arange(0, self.volumeLength)
        return (x, y, z)

    """
    Returns 3D matrix of grey colors representing the values of the image volume
    Used for creating interpolated images
    """

    def getValues(self):
        values = []
        for i in range(len(self.images)):
            input_img_raw_string = '{}/{}'.format(self._folder_path, self.images[i])
            twoD_array = [*qimage2ndarray.imread(input_img_raw_string)]
            values.append(twoD_array)
        return np.array(np.transpose(values))

    """
    Returns 3D matrix of 0s and 1s representing annotated ground truth
    Used for creating interpolated images
    """

    def getValuesGT(self):
        values = []
        for i in range(len(self.imagesGT)):
            input_img_raw_string = '{}/{}'.format(self._folder_path, self.imagesGT[i])
            twoD_array = [*qimage2ndarray.imread(input_img_raw_string, masked=True)]
            values.append(twoD_array)
        return np.array(values)

    """
    Returns matrix of points on viewing plane
    Used for creating interpolated images
    """

    def get3DCoords(self):
        row_range = np.arange(-self.width_Scaled / 2, self.width_Scaled / 2, self.width_Scaled / self.windowWidth)
        col_range = np.arange(-self.height_Scaled / 2, self.height_Scaled / 2, self.height_Scaled / self.windowHeight)
        [u, v] = np.meshgrid(row_range, col_range)
        u = u.flatten()
        v = v.flatten()
        u = u.reshape(u.size, 1)
        v = v.reshape(v.size, 1)
        result = self.c + u * self.v1 + v * self.v2
        return result

    """
    Sets variables saved in glWidget instance
    """
    def setGLVariables(self,coords):
        self.glWidget.v1 = self.v1
        self.glWidget.v2 = self.v2
        self.glWidget.n = self.n
        self.glWidget.setC(self.c)
        self.glWidget.width_Scaled=self.width_Scaled
        self.glWidget.height_Scaled=self.height_Scaled
        self.glWidget.p = coords[0]
        corner2 = coords[self.windowWidth-1]
        corner3 = coords[(self.windowWidth-1) * self.windowWidth]
        corner4 = coords[len(coords) - 1]
        zTranslate = 1/2 - (self.volumeLength/(self.volumeWidth*2))
        self.glWidget.corners = [
            self.glWidget.p / self.volumeWidth + np.array([0, 0, zTranslate]),
            corner2 / self.volumeWidth + np.array([0, 0, zTranslate]),
            corner4 / self.volumeWidth + np.array([0, 0, zTranslate]),
            corner3 / self.volumeWidth + np.array([0, 0, zTranslate])]

    """
    Finds points where the viewing plane intersects with skeletons of branches
    """
    def getIntersections(self):
        intersections = []
        c = 0
        for branch in self.glWidget.skeleton:
            if len(branch)>1:
                for i in range(len(branch)-1):
                    p = np.array(branch[i])
                    q = np.array(branch[i+1])
                    h1 = np.dot(p-self.glWidget.c, self.n)
                    h2 = np.dot(q-self.glWidget.c, self.n)
                    if h1*h2 < 0:
                        zTranslate = 1 / 2 - (self.volumeLength / (self.volumeWidth * 2))
                        R = p + (q-p) * (h1/(h1-h2))
                        R = (R - np.array([0, 0, zTranslate]))*self.volumeWidth
                        xGlob = 400 + np.dot(R-self.c, self.v1)
                        yGlob = 400 + np.dot(R-self.c, self.v2)
                        corner = 400 - self.width_Scaled/2
                        x = (xGlob-corner) * self.windowWidth/self.width_Scaled
                        y = (yGlob-corner) * self.windowWidth/self.width_Scaled
                        intersections.append([x,y,self.glWidget.polygon_colors[c]])
            c += 1
        return intersections

    """
    Based on current v1,v2,n, and c, creates interpolated image and updates 2d window
    """
    def showPlane(self):
        self.v1 = self.view.v1
        self.v2 = self.view.v2
        self.n = np.cross(self.view.v1,self.view.v2)
        self.c = self.view.c
        coords = self.get3DCoords()
        # update plane for glWidget so that 3d window updates
        self.setGLVariables(coords)

        if self.loadImgs == True:
            colors = self.interpolate(coords)
            colors = colors.reshape((self.windowWidth, self.windowHeight))
            image = qimage2ndarray.array2qimage(colors)
            for i in range(self.windowWidth):
                image.setPixelColor(i, 0, QColor(255, 0, 0))
                image.setPixelColor(i, 1, QColor(255, 0, 0))
                image.setPixelColor(i, 2, QColor(255, 0, 0))
            for j in range(self.windowHeight):
                image.setPixelColor(0, j, QColor(0, 255, 0))
                image.setPixelColor(1, j, QColor(0, 255, 0))
                image.setPixelColor(2, j, QColor(0, 255, 0))
            pixmap = QPixmap.fromImage(image)
            if self.showGT_img == False:
                self.view.pixmap.setPixmap(pixmap)
            self.view.drawIntersections(self.getIntersections())


def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
