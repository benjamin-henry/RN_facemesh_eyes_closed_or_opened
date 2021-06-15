import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { LogBox, StyleSheet, View, Text} from 'react-native';
import Svg, { Circle, Rect, G, Line} from 'react-native-svg';

import * as tf from "@tensorflow/tfjs"

import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection"

import { Camera } from 'expo-camera';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';

const TensorCamera = cameraWithTensors(Camera);
let model = null;
LogBox.ignoreAllLogs(true)


function euclidean_dist (x1, y1, x2, y2) {
  return Math.sqrt( Math.pow((x1-x2), 2) + Math.pow((y1-y2), 2) );
};

let textureDims;
    if (Platform.OS === 'ios') {
     textureDims = {
       height: 1920,
       width: 1080,
     };
    } else {
     textureDims = {
       height: 1600,
       width: 1200,
     };
    }

export default class App extends React.Component {

  constructor(props) {
    super(props)
    this.state= {
      tfready:false,
      cameraType: Camera.Constants.Type.front,
      hasPermission:null,
      faceDetector:null,
      enableDetections: true,
      faces: null,
      eyesState: []
    }
    this.handleCameraStream = this.handleCameraStream.bind(this);
  }

  async componentDidMount() {
    const { status } = await Camera.requestPermissionsAsync();

    this.setState({hasPermission: status==='granted'})
    
    await tf.ready();
    // await tf.setBackend("rn-webgl");
    
    model = await faceLandmarksDetection.load(
      faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
      {shouldLoadIrisModel:true});
      
    this.setState({faceDetector: model})
    this.setState({tfready:true});
  }

  componentWillUnmount() {
    if(this.rafID) {
      cancelAnimationFrame(this.rafID);
    }
  }

  async handleCameraStream(images, updatePreview, gl) {
    const loop = async () => {
      const nextImageTensor = images.next().value;
      if(this.state.faceDetector!=null && this.state.enableDetections===true) {
        const preds = await this.state.faceDetector.estimateFaces({input:nextImageTensor,returnTensors:false})
        this.setState({faces:preds})
        preds.forEach((face) => {
          const leftEyeLower = face["annotations"]["leftEyeLower0"]
          const leftEyeUpper = face["annotations"]["leftEyeUpper0"]
          
          const leftCenterLower = leftEyeLower[4]
          const leftCenterUpper = leftEyeUpper[4]
          const leftLeft = leftEyeLower[0]
          const leftRight = leftEyeLower[8]

          const leftVertDist = euclidean_dist(leftCenterLower[0],leftCenterLower[1],leftCenterUpper[0],leftCenterUpper[1])
          const leftHorizDist = euclidean_dist(leftLeft[0],leftLeft[1],leftRight[0],leftRight[1])
          const leftClosedScore = leftVertDist / (2.*leftHorizDist)         
          
          const rightEyeLower = face["annotations"]["rightEyeLower0"]
          const rightEyeUpper = face["annotations"]["rightEyeUpper0"]
          
          const rightCenterLower = rightEyeLower[4]
          const rightCenterUpper = rightEyeUpper[4]
          const rightLeft = rightEyeLower[0]
          const rightRight = rightEyeLower[8]

          const rightVertDist = euclidean_dist(rightCenterLower[0],rightCenterLower[1],rightCenterUpper[0],rightCenterUpper[1])
          const rightHorizDist = euclidean_dist(rightLeft[0],rightLeft[1],rightRight[0],rightRight[1])
          const rightClosedScore = rightVertDist / (2.*rightHorizDist)

          const leftClosed = leftClosedScore < .1 ? "closed" : "opened";
          const rightClosed = rightClosedScore < .1 ? "closed" : "opened";    

          this.setState({eyesState:[rightClosed, leftClosed]})
          // console.log(leftClosedScore, rightClosedScore)

        })    
      } 
      tf.dispose(nextImageTensor)     
      this.rafID = requestAnimationFrame(loop);
    }
    loop();
  }

  renderScores() {
    if(this.state.scores != null) {
      return null
    }
  }

  renderFaces() {
  const {faces} = this.state;
    if(faces != null) {
      const faceBoxes = faces.map((f, fIndex) => {
        const topLeft = f["boundingBox"].topLeft;
        const bottomRight = f["boundingBox"].bottomRight;
        const landmarks1 = (f["annotations"]["leftEyeLower0"]).map((l, lIndex) => {
          return <Circle
            key={`landmark_${fIndex}_${lIndex}`}
            cx={l[0]}
            cy={l[1]}
            r='2'
            strokeWidth='0'
            fill='blue'
            />;
        });
        const landmarks2 = (f["annotations"]["leftEyeUpper0"]).map((l, lIndex) => {
          return <Circle
            key={`landmark_${fIndex}_${lIndex}`}
            cx={l[0]}
            cy={l[1]}
            r='2'
            strokeWidth='0'
            fill='blue'
            />;
        });
        const landmarks3 = (f["annotations"]["rightEyeLower0"]).map((l, lIndex) => {
          return <Circle
            key={`landmark_${fIndex}_${lIndex}`}
            cx={l[0]}
            cy={l[1]}
            r='2'
            strokeWidth='0'
            fill='blue'
            />;
        });
        const landmarks4 = (f["annotations"]["rightEyeUpper0"]).map((l, lIndex) => {
          return <Circle
            key={`landmark_${fIndex}_${lIndex}`}
            cx={l[0]}
            cy={l[1]}
            r='2'
            strokeWidth='0'
            fill='blue'
            />;
        });

        return <G key={`facebox_${fIndex}`}>
          <Rect
            x={topLeft[0]}
            y={topLeft[1]}
            fill={'red'}
            fillOpacity={0.2}
            width={(bottomRight[0] - topLeft[0])}
            height={(bottomRight[1] - topLeft[1])}
          />
          {landmarks1}
          {landmarks2}
          {landmarks3}
          {landmarks4}
        </G>;
      });

      const flipHorizontal = Platform.OS === 'ios' ? 1 : -1;
      return <Svg height='100%' width='100%'
        viewBox={`0 0 ${400} ${300}`}
        scaleX={flipHorizontal}>
          {faceBoxes}
        </Svg>;
    } else {
      return null;
    }
  }


  render() {
    // Currently expo does not support automatically determining the
    // resolution of the camera texture used. So it must be determined
    // empirically for the supported devices and preview size. 
    
 
    return (
    <View style={styles.container}>
      {this.state.tfready && (
        <>
        <React.Fragment>
          <TensorCamera
          // Standard Camera props
          style={styles.camera}
          type={this.state.cameraType}
          // Tensor related props
          cameraTextureHeight={textureDims.height}
          cameraTextureWidth={textureDims.width}
          resizeHeight={400}
          resizeWidth={300}
          resizeDepth={3}
          onReady={this.handleCameraStream}
          autorender={true}
          />
          <View style={styles.facesResults}>
              {/* {this.renderFaces()} */}
          </View>
        </React.Fragment>
        <View style={{position:"absolute", top:540, width:300,alignItems:"center", justifyContent:"center", flexDirection:"row",flex:1}}>
          <View style={{flex:1, justifyContent:"center", alignItems:"center"}}>
            <Text style={{color:"white", fontSize:20,flex:1}}>{this.state.eyesState[1]}</Text>
          </View>
          <View style={{flex:1, justifyContent:"center", alignItems:"center"}}>
          <Text style={{color:"white", fontSize:20,flex:1}}>{this.state.eyesState[0]}</Text>
          </View>
        
        
        </View>
        
        </>
      )}
    </View>)
   }
}

const styles = StyleSheet.create({
  container: {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    height: '100%',
    backgroundColor: '#000',
  },
  camera : {
    position:'absolute',
    top:100,
    width: 300,
    height: 400,
    zIndex: 1,
    borderWidth: 1,
    borderColor: 'white',
    borderRadius: 0,
  },
  facesResults: {
    position:'absolute',
    top:100,
    width: 300,
    height: 400,
    zIndex: 20,
    borderWidth: 1,
    borderColor: 'white',
    borderRadius: 0,
  },
});
