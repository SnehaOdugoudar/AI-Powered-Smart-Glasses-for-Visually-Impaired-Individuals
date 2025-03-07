#include "esp_camera.h"
#include "WiFi.h"
#include "ESPAsyncWebServer.h"

#define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM
#include "camera_pins.h"

// Replace with your Wi-Fi credentials
const char* ssid = "Orbic-Speed-RC400L-wxyz";
const char* password = "z0ruicrhrm";

AsyncWebServer server(80);

// void startCameraServer() {
//     server.on("/stream", HTTP_GET, [](AsyncWebServerRequest *request){
//         camera_fb_t *fb = esp_camera_fb_get();
//         if (!fb) {
//             request->send(500, "text/plain", "Camera frame capture failed");
//             return;
//         }
//         AsyncWebServerResponse *response = request->beginResponse(200, "image/jpeg", fb->buf, fb->len); // Send as plain JPEG
//         response->addHeader("Access-Control-Allow-Origin", "*");
//         response->addHeader("Cache-Control", "no-store");
//         request->send(response);
//         esp_camera_fb_return(fb);
//     });
//     server.begin();
// }

void startCameraServer() {
     server.on("/mjpeg", HTTP_GET, [](AsyncWebServerRequest *request) {
        AsyncWebServerResponse *response = request->beginChunkedResponse("multipart/x-mixed-replace; boundary=frame", [](uint8_t *buffer, size_t maxLen, size_t index) -> size_t {
            Serial.println("Attempting to capture frame..."); // Added debug print
            camera_fb_t *fb = esp_camera_fb_get();
            if (!fb) {
                Serial.println("Camera frame capture failed!"); // Error during capture
                return 0; // Indicate error
            }
            Serial.printf("Frame captured, length: %u bytes\n", fb->len); // Frame capture success
            size_t fbLen = fb->len;
            if (fbLen > maxLen) {
                fbLen = maxLen;
            }
            memcpy(buffer, fb->buf, fbLen);
            esp_camera_fb_return(fb);
            return fbLen;
        });
        response->addHeader("Cache-Control", "no-cache");
        response->addHeader("Connection", "keep-alive");
        request->send(response);
    });

    server.begin();
}

void setup() {
    Serial.begin(115200);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("WiFi connected");

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.frame_size = FRAMESIZE_QQVGA;
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality = 40;
    config.fb_count = 2;

    if (psramFound()) {
        config.jpeg_quality = 20;
    } else {
        config.frame_size = FRAMESIZE_QQVGA;
        config.fb_location = CAMERA_FB_IN_DRAM;
    }

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera init failed");
        return;
    }

    Serial.println("Camera ready");
    Serial.println(WiFi.localIP());

    startCameraServer();
}

void loop() {
    delay(100);
}
