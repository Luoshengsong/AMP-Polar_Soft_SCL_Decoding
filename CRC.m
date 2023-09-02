classdef CRC
    properties
        CRCbit_Len
        polynomial
        CRCgen
        CRCdet
    end
    
    methods
        function obj = CRC(CRCbit_Len) %(polynomial)
            obj.CRCbit_Len = CRCbit_Len;
            switch CRCbit_Len
                case 0
                    obj.polynomial = [1];
                case 1
                    obj.polynomial = [1 1];
                case 2
                    obj.polynomial = [1 1 1];
                case 3
                    obj.polynomial = [1 1 1 1];
                case 4
                    obj.polynomial = [1 0 0 1 1]; % x^4 + x + 1
                case 5
                    obj.polynomial = [1 0 0 1 0 1];
                case 6
                    obj.polynomial = [1 0 0 0 0 1 1];  %1000001
                case 7
                    obj.polynomial = [1 0 0 0 1 1 1 1];
                case 8
                    obj.polynomial = [1 0 0 1 1 0 0 0 1];
                case 9
                    obj.polynomial = [1 0 0 0 1 0 0 0 0 1];
                case 10
                    obj.polynomial = [1 0 0 0 0 1 0 0 0 1 1];
                case 11
                    obj.polynomial = [1 0 0 0 0 0 0 1 0 0 0 1];
                case 12
                    obj.polynomial = [1 0 0 1 0 0 0 0 0 0 0 1 1];
                case 13
                    obj.polynomial = [1 0 0 0 1 0 0 0 0 0 0 0 1 1]; 
                case 14
                    obj.polynomial = [1 0 0 0 0 1 0 1 0 0 0 0 0 1 1];
                case 15
                    obj.polynomial = [1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1];
                case 16
                    obj.polynomial = [1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1];
                case 24
                    obj.polynomial = [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1];
                otherwise
                    disp(["Please add the corresponding polynomial in binary form."]);
            end
                obj.CRCgen = crc.generator('Polynomial', obj.polynomial, ...
                                   'InitialState', zeros(1, obj.CRCbit_Len), ...
                                   'FinalXOR', zeros(1, obj.CRCbit_Len));
                obj.CRCdet = crc.detector('Polynomial', obj.polynomial, ...
                                   'InitialState', zeros(1, obj.CRCbit_Len), ...
                                   'FinalXOR', zeros(1, obj.CRCbit_Len));
            
        end
        
        function crcencoded_data = encode(obj, data)
            crcencoded_data = generate(obj.CRCgen, data); % SourceCoding_Len x No_Active_Users
        end
        
        function [crcdecoded_data, error_detected] = decode(obj, received_data)
            [crcdecoded_data, error_detected] = detect(obj.CRCdet, received_data); 
        end
        

    end
end

