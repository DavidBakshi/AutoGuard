import React, { useEffect, useState } from 'react';
import { SafeAreaView, StyleSheet, Text, View, Button, Alert } from 'react-native';
import MapView, { Marker } from 'react-native-maps';

const API_URL = 'http://10.0.2.2:5000'; // Use your computer's IP mapping for the emulator

const App = () => {
  const [lastPoint, setLastPoint] = useState<{ latitude: number, longitude: number } | null>(null);
  const [notification, setNotification] = useState<string>('');
  const [latestObd, setLatestObd] = useState<any>(null);

  // Function to fetch the last coordinate
  const fetchLastPoint = async () => {
    try {
      const response = await fetch(`${API_URL}/last_point`);
      const data = await response.json();
      if (data.geometry && data.geometry.coordinates) {
        const [longitude, latitude] = data.geometry.coordinates;
        setLastPoint({ latitude, longitude });
      }
    } catch (error) {
      console.error("Error fetching last point:", error);
    }
  };

  // Function to send the kill switch command
  const sendKillSwitch = async () => {
    try {
      const response = await fetch(`${API_URL}/kill_switch`, { method: 'POST' });
      const data = await response.json();
      setNotification(data.notification);
      Alert.alert("Kill Switch", data.notification);
    } catch (error) {
      console.error("Error sending kill switch command:", error);
    }
  };

  // Function to poll for latest OBD data
  const pollObdData = async () => {
    try {
      const response = await fetch(`${API_URL}/latest_obd`);
      const data = await response.json();
      if (data.data) {
        // If new OBD data is available, set it as a notification
        setLatestObd(data);
        setNotification("New OBD data received. Review details and decide on kill switch.");
        Alert.alert("OBD Data", "New OBD data received. Please check details and decide if you want to activate the kill switch.");
      }
    } catch (error) {
      console.error("Error polling OBD data:", error);
    }
  };

  useEffect(() => {
    // Fetch the last coordinate when the app loads
    fetchLastPoint();

    // Start polling for OBD data every 5 seconds (adjust as needed)
    const intervalId = setInterval(pollObdData, 5000);
    return () => clearInterval(intervalId);
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.header}>Auto-Guard</Text>
      {lastPoint ? (
        <MapView
          style={styles.map}
          initialRegion={{
            latitude: lastPoint.latitude,
            longitude: lastPoint.longitude,
            latitudeDelta: 0.01,
            longitudeDelta: 0.01,
          }}
        >
          <Marker coordinate={lastPoint} title="Last Location" />
        </MapView>
      ) : (
        <Text>Loading last coordinate...</Text>
      )}
      <View style={styles.buttonContainer}>
        <Button title="Activate Kill Switch" onPress={sendKillSwitch} />
      </View>
      {notification ? <Text style={styles.notification}>{notification}</Text> : null}
      {latestObd && (
        <View style={styles.obdContainer}>
          <Text style={styles.header}>Latest OBD Data:</Text> // is this should be obdHeader?
          <Text>{JSON.stringify(latestObd.data)}</Text>
        </View>
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16, backgroundColor: '#fff' },
  header: { fontSize: 24, fontWeight: 'bold', marginBottom: 16, textAlign: 'center' },
  map: { width: '100%', height: 300, marginBottom: 16 },
  buttonContainer: { marginVertical: 8 },
  notification: { marginTop: 16, textAlign: 'center', color: 'red' },
  obdContainer: { marginTop: 16, padding: 8, backgroundColor: '#eee', borderRadius: 8 }
});

export default App;
