"""FX150 USB 디스크립터 덤프 (읽기 전용).

VID_34DB/PID_8004의 인터페이스/엔드포인트 구조를 확인한다.
제어가 어느 인터페이스(오디오/HID/벤더)로 가는지 파악해 프로토콜 RE 방향을 정한다.

주의: FLAMMA 에디터가 실행 중이면 인터페이스를 점유해 일부 정보가 안 보일 수 있음.
"""
import usb.core
import usb.backend.libusb1

VID, PID = 0x34DB, 0x8004
DLL = r"C:\Program Files (x86)\FLAMMA\FX150\libusb-1.0.dll"

CLASS_NAMES = {
    0x00: "Device", 0x01: "Audio", 0x02: "CDC", 0x03: "HID",
    0x08: "MassStorage", 0x0A: "CDC-Data", 0x0B: "SmartCard",
    0xFE: "App-Specific", 0xFF: "Vendor-Specific",
}


def main():
    be = usb.backend.libusb1.get_backend(find_library=lambda x: DLL)
    if be is None:
        print("libusb 백엔드 로드 실패."); return
    dev = usb.core.find(idVendor=VID, idProduct=PID, backend=be)
    if dev is None:
        print(f"USB {VID:#06x}:{PID:#06x} 미발견."); return

    print(f"장치 {VID:#06x}:{PID:#06x}  bDeviceClass={dev.bDeviceClass:#04x} "
          f"({CLASS_NAMES.get(dev.bDeviceClass,'?')})")
    try:
        print(f"제조사={usb.util.get_string(dev, dev.iManufacturer)}  "
              f"제품={usb.util.get_string(dev, dev.iProduct)}")
    except Exception as e:
        print(f"문자열 읽기 실패: {e}")

    for cfg in dev:
        print(f"\nConfiguration {cfg.bConfigurationValue}: {cfg.bNumInterfaces} interfaces")
        for intf in cfg:
            cls = intf.bInterfaceClass
            print(f"  Interface {intf.bInterfaceNumber} alt {intf.bAlternateSetting}: "
                  f"class={cls:#04x} ({CLASS_NAMES.get(cls,'?')}) "
                  f"subclass={intf.bInterfaceSubClass:#04x} proto={intf.bInterfaceProtocol:#04x} "
                  f"endpoints={intf.bNumEndpoints}")
            for ep in intf:
                ep_dir = "IN" if usb.util.endpoint_direction(ep.bEndpointAddress) else "OUT"
                ep_type = {0: "control", 1: "iso", 2: "bulk", 3: "interrupt"}[
                    usb.util.endpoint_type(ep.bmAttributes)]
                print(f"      EP {ep.bEndpointAddress:#04x} {ep_dir:3} {ep_type:9} "
                      f"maxpkt={ep.wMaxPacketSize}")


if __name__ == "__main__":
    main()
