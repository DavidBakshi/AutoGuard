import React, { useState, useEffect } from "react";
import { Platform, View, Text, StyleSheet, SafeAreaView } from "react-native";
import MapView, { Marker } from "react-native-maps";

export default function App() {
  const [coordinates, setCoordinates] = useState({
    latitude: 31.77759352590111,
    longitude: 35.197599143135946,
    latitudeDelta: 0.05,
    longitudeDelta: 0.05,
  });
  const [obdStatus, setObdStatus] = useState({
    detected: false,
    time: null,
  });
  // local UI flag to control banner visibility
  const [showBanner, setShowBanner] = useState(false);

  const BASE_URL =
    Platform.OS === "android"
      ? "http://10.0.2.2:5000"
      : "http://localhost:5000";

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const locRes = await fetch(`${BASE_URL}/current_location`);
        const locData = await locRes.json();
        setCoordinates(locData);
      } catch (e) {
        console.error("Failed to fetch coordinates:", e);
      }
      try {
        const obdRes = await fetch(`${BASE_URL}/obd_detected`);
        const obdData = await obdRes.json();
        setObdStatus(obdData);
      } catch (e) {
        console.error("Failed to fetch OBD status:", e);
      }
    };

    fetchAll();
    const interval = setInterval(fetchAll, 1000);
    return () => clearInterval(interval);
  }, []);

  // whenever server reports a new detection time, flash the banner
  useEffect(() => {
    if (obdStatus.detected) {
      setShowBanner(true);
      const t = setTimeout(() => setShowBanner(false), 5000);
      return () => clearTimeout(t);
    }
  }, [obdStatus.time]);

  return (
    <SafeAreaView style={styles.container}>
      <MapView style={styles.map} region={coordinates}>
        <Marker
          coordinate={coordinates}
          title="Your Location"
          description="This is the selected location."
        />
      </MapView>

      {showBanner && (
        <View style={styles.banner}>
          <Text style={styles.warningText}>
            ⚠️ OBD Detected at {obdStatus.time}
          </Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  map: {
    flex: 1,
  },
  banner: {
    position: "absolute",
    top: "45%", // vertically centered
    left: 20,
    right: 20,
    backgroundColor: "#ffdddd",
    padding: 16,
    borderRadius: 8,
    alignItems: "center",
    elevation: 10, // Android shadow
    shadowColor: "#000", // iOS shadow
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  warningText: {
    color: "#990000",
    fontSize: 18,
    fontWeight: "bold",
  },
});
