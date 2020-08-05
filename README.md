# Writing-in-Air

<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:ugcPost:6664606953697882114" height="763" width="504" frameborder="0" allowfullscreen="" title="Embedded post"></iframe>

![writing_in_air](https://user-images.githubusercontent.com/68480967/89462039-76fd6c80-d78a-11ea-924c-9285344df3f6.gif)

> Here in this program you can draw numbers in the air, and the trained neural network is used to identify what we drew. 

> The program was specifically configured to detect the tip of my blue pen.

> Numbers can be configured using the "color_selector.py" file.

> Each color lies in the specific HSV value range, hence taking advantage of this we can seperate any color from background.

> Once the color is seperated, we can mark that point on the live video stream.

> By pressing SPACEBAR we can start traking the points [ i.e. we can start drawing and it will stay ]

> Once our drawing is completed [ drawing a number ], we can press Spacebar again to dave our drawing.

> And then immediately the drawing is passed to our model to predict the number.
