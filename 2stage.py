import numpy as np
from PIL import Image
import os
import random

class SteganographyMultiLevel:
    def __init__(self, block_size=32, num_blocks=5):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.end_marker = [0, 0, 0, 0, 0, 0, 0, 0]  # 8 нулевых битов как маркер конца
        
        # Расчет общей вместимости для всех уровней
        self.pixels_per_block = block_size * block_size  # 1024
        self.total_pixels = self.pixels_per_block * num_blocks  # 5120
        self.total_channels = 3  # RGB
        
        # Вместимость для каждого уровня битности
        self.capacities = {}
        for n_bits in range(1, 9):
            total_bits = self.total_pixels * self.total_channels * n_bits
            self.capacities[n_bits] = {
                'total_bits': total_bits,
                'total_bytes': total_bits // 8,
                'total_chars': (total_bits // 8) - 1  # минус маркер конца
            }
    
    def calculate_entropy(self, block):
        """Вычисление энтропии Шеннона для блока"""
        flat_block = block.flatten()
        hist = np.bincount(flat_block, minlength=256)
        hist = hist[hist > 0] / len(flat_block)
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    def find_high_entropy_blocks(self, image_array):
        """Поиск 5 блоков с максимальной энтропией"""
        h, w = image_array.shape[:2]
        block_entropies = []
        
        # Перебираем все возможные блоки 32x32
        for y in range(0, h - self.block_size + 1, self.block_size):
            for x in range(0, w - self.block_size + 1, self.block_size):
                block = image_array[y:y+self.block_size, x:x+self.block_size]
                
                # Для цветного изображения усредняем энтропию по каналам
                if len(image_array.shape) == 3:
                    entropies = [self.calculate_entropy(block[:,:,c]) for c in range(3)]
                    entropy = np.mean(entropies)
                else:
                    entropy = self.calculate_entropy(block)
                
                block_entropies.append(((x, y), entropy))
        
        # Сортируем по убыванию энтропии и берем первые 5 блоков
        block_entropies.sort(key=lambda x: x[1], reverse=True)
        selected_blocks = [pos for pos, _ in block_entropies[:self.num_blocks]]
        
        print(f"Найдено 5 блоков 32x32 с максимальной энтропией:")
        for i, (x, y) in enumerate(selected_blocks, 1):
            print(f"  Блок {i}: ({x}, {y}), энтропия = {block_entropies[i-1][1]:.4f}")
        
        return selected_blocks
    
    def create_mask(self, image_shape, selected_blocks):
        """Создание маски с отмеченными блоками - ВСЕ 5 блоков белые"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        for x, y in selected_blocks:
            mask[y:y+self.block_size, x:x+self.block_size] = 255
            
        return mask
    
    def text_to_bits(self, text):
        """Преобразование текста в биты"""
        bits = []
        for char in text:
            char_bits = format(ord(char), '08b')
            bits.extend([int(b) for b in char_bits])
        return bits
    
    def bits_to_text(self, bits):
        """Преобразование битов обратно в текст"""
        chars = []
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte = bits[i:i+8]
                char_code = int(''.join(map(str, byte)), 2)
                if 32 <= char_code <= 126:  # Только печатные символы
                    chars.append(chr(char_code))
        return ''.join(chars)
    
    def generate_random_bits(self, count):
        """Генерация случайных битов для заполнения"""
        return [random.randint(0, 1) for _ in range(count)]
    
    def embed_message_nbits_full(self, image_array, message, selected_blocks, n_bits):
        """
        ПОЛНОЕ заполнение всех 5 блоков информацией
        Сообщение + случайные биты до полного заполнения
        """
        img_copy = image_array.copy()
        
        # Проверяем, что изображение цветное
        if len(img_copy.shape) != 3 or img_copy.shape[2] != 3:
            raise ValueError("Изображение должно быть цветным (RGB) с 3 каналами")
        
        # Получаем вместимость для данного уровня
        capacity = self.capacities[n_bits]
        total_bits_needed = capacity['total_bits']
        
        # Преобразуем сообщение в биты
        message_bits = self.text_to_bits(message)
        
        # Добавляем маркер конца
        message_with_marker = message_bits + self.end_marker
        message_len = len(message_with_marker)
        
        print(f"\n  ПОЛНОЕ ЗАПОЛНЕНИЕ с {n_bits} бит/пиксель:")
        print(f"    Размер блока: {self.block_size}x{self.block_size} = {self.pixels_per_block} пикселей")
        print(f"    Всего блоков: {self.num_blocks}")
        print(f"    Всего пикселей: {self.total_pixels}")
        print(f"    Каналов на пиксель: {self.total_channels}")
        print(f"    ВСЕГО ДОСТУПНО БИТ: {total_bits_needed}")
        print(f"    Максимум символов: {capacity['total_chars']}")
        print(f"    Битов в сообщении: {len(message_bits)}")
        print(f"    Битов с маркером: {message_len}")
        
        # Генерируем случайные биты для заполнения оставшегося места
        if message_len < total_bits_needed:
            random_bits_needed = total_bits_needed - message_len
            random_bits = self.generate_random_bits(random_bits_needed)
            all_bits = message_with_marker + random_bits
            print(f"    Добавлено случайных бит: {random_bits_needed}")
        else:
            all_bits = message_with_marker[:total_bits_needed]
            print(f"    ВНИМАНИЕ: Сообщение обрезано до {total_bits_needed} бит")
        
        print(f"    ВСЕГО БИТ ДЛЯ ВСТРАИВАНИЯ: {len(all_bits)}")
        
        # Встраиваем биты последовательно во все пиксели всех блоков
        bit_index = 0
        total_pixels_modified = 0
        changes_log = []
        
        for block_idx, (block_x, block_y) in enumerate(selected_blocks):
            print(f"    Блок {block_idx+1}: ({block_x}, {block_y})")
            pixels_in_block = 0
            
            for y in range(block_y, block_y + self.block_size):
                for x in range(block_x, block_x + self.block_size):
                    if bit_index >= len(all_bits):
                        break
                    
                    # Для каждого пикселя изменяем n бит в каждом канале
                    for c in range(3):  # R, G, B каналы
                        if bit_index >= len(all_bits):
                            break
                        
                        # Текущее значение пикселя
                        current_value = img_copy[y, x, c]
                        
                        # Берем следующие n бит из all_bits
                        bits_to_embed = 0
                        for b in range(n_bits):
                            if bit_index < len(all_bits):
                                if all_bits[bit_index] == 1:
                                    bits_to_embed |= (1 << b)
                                bit_index += 1
                        
                        # Очищаем n младших бит и вставляем новые
                        mask = (1 << n_bits) - 1
                        new_value = (current_value & ~mask) | bits_to_embed
                        
                        # Логируем первые несколько изменений для отладки
                        if len(changes_log) < 5:
                            old_bits = current_value & mask
                            changes_log.append(f"        ({x},{y},{c}): {current_value:3d} -> {new_value:3d} (биты: {bin(old_bits)[2:].zfill(n_bits)} -> {bin(bits_to_embed)[2:].zfill(n_bits)})")
                        
                        # Применяем изменения
                        img_copy[y, x, c] = new_value
                    
                    pixels_in_block += 1
                    total_pixels_modified += 1
                    
                if bit_index >= len(all_bits):
                    break
            
            print(f"      Изменено пикселей в блоке: {pixels_in_block}")
        
        # Выводим несколько примеров изменений
        if changes_log:
            print(f"    Примеры изменений (первые 5):")
            for log in changes_log:
                print(log)
        
        print(f"    ВСЕГО ИЗМЕНЕНО ПИКСЕЛЕЙ: {total_pixels_modified}")
        print(f"    ВСЕГО ИСПОЛЬЗОВАНО БИТ: {bit_index}")
        print(f"    Заполнение: {bit_index}/{total_bits_needed} бит ({bit_index/total_bits_needed*100:.1f}%)")
        
        return img_copy
    
    def extract_message_nbits_full(self, image_array, selected_blocks, n_bits):
        """
        Извлечение ВСЕХ битов из 5 блоков
        """
        capacity = self.capacities[n_bits]
        total_bits_to_extract = capacity['total_bits']
        extracted_bits = []
        bits_extracted = 0
        
        print(f"\n  Извлечение ВСЕХ битов с {n_bits} бит/пиксель:")
        print(f"    Всего бит для извлечения: {total_bits_to_extract}")
        
        for block_idx, (block_x, block_y) in enumerate(selected_blocks):
            for y in range(block_y, block_y + self.block_size):
                for x in range(block_x, block_x + self.block_size):
                    for c in range(3):
                        if bits_extracted >= total_bits_to_extract:
                            break
                            
                        # Извлекаем n бит из пикселя
                        pixel_value = image_array[y, x, c]
                        mask = (1 << n_bits) - 1
                        pixel_bits = pixel_value & mask
                        
                        # Извлекаем каждый бит
                        for b in range(n_bits):
                            if bits_extracted >= total_bits_to_extract:
                                break
                            bit = (pixel_bits >> b) & 1
                            extracted_bits.append(bit)
                            bits_extracted += 1
                    
                    if bits_extracted >= total_bits_to_extract:
                        break
                if bits_extracted >= total_bits_to_extract:
                    break
            if bits_extracted >= total_bits_to_extract:
                break
        
        print(f"    Извлечено бит: {bits_extracted}")
        
        # Ищем маркер конца, чтобы найти сообщение
        message_bits = []
        for i in range(len(extracted_bits) - 7):
            if extracted_bits[i:i+8] == self.end_marker:
                message_bits = extracted_bits[:i]
                print(f"    Найден маркер конца на позиции {i}")
                print(f"    Битов в сообщении: {len(message_bits)}")
                break
        else:
            message_bits = extracted_bits
            print(f"    Маркер конца не найден, возвращаем все биты")
        
        return message_bits, extracted_bits
    
    def process_all_levels_full(self, image_path, message, output_dir="stego_output"):
        """
        Создает 8 изображений с ПОЛНЫМ заполнением 5 блоков
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Загружаем изображение
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        print("=" * 90)
        print("СТЕГАНОГРАФИЯ: ПОЛНОЕ ЗАПОЛНЕНИЕ 5 БЛОКОВ 32x32")
        print("=" * 90)
        print(f"Изображение: {image_path}")
        print(f"Размер: {img_array.shape[1]}x{img_array.shape[0]}")
        print(f"Сообщение: '{message}'")
        print(f"Длина сообщения: {len(message)} символов = {len(message) * 8} бит")
        print("-" * 90)
        
        # Находим 5 блоков с максимальной энтропией
        selected_blocks = self.find_high_entropy_blocks(img_array)
        
        # Создаем маску (ВСЕ 5 блоков белые)
        mask = self.create_mask(img_array.shape, selected_blocks)
        mask_img = Image.fromarray(mask)
        mask_path = os.path.join(output_dir, "mask.jpg")
        mask_img.save(mask_path)
        print(f"\nМаска сохранена: {mask_path}")
        print("-" * 90)
        
        # Сохраняем оригинал для сравнения
        original_path = os.path.join(output_dir, "original.jpg")
        img.save(original_path)
        
        results = {}
        
        # Для каждого уровня битности (1-8)
        for n_bits in range(1, 9):
            print(f"\n{'='*70}")
            print(f"УРОВЕНЬ {n_bits} БИТ НА ПИКСЕЛЬ - ПОЛНОЕ ЗАПОЛНЕНИЕ")
            print(f"{'='*70}")
            
            try:
                capacity = self.capacities[n_bits]
                print(f"Вместимость: {capacity['total_chars']} символов, {capacity['total_bits']} бит")
                
                # Встраиваем сообщение с ПОЛНЫМ заполнением
                embedded_array = self.embed_message_nbits_full(
                    img_array.copy(), message, selected_blocks, n_bits
                )
                
                # Сохраняем изображение
                output_filename = f"embedded_{n_bits}bits_full.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                embedded_img = Image.fromarray(embedded_array)
                embedded_img.save(output_path)
                
                # Извлекаем ВСЕ биты для проверки
                extracted_message_bits, all_extracted_bits = self.extract_message_nbits_full(
                    embedded_array, selected_blocks, n_bits
                )
                extracted_message = self.bits_to_text(extracted_message_bits)
                
                # Подсчитываем изменения
                changes = 0
                for bx, by in selected_blocks:
                    for y in range(by, by + self.block_size):
                        for x in range(bx, bx + self.block_size):
                            if not np.array_equal(img_array[y, x], embedded_array[y, x]):
                                changes += 1
                
                results[n_bits] = {
                    'path': output_path,
                    'extracted': extracted_message,
                    'correct': extracted_message == message,
                    'changed_pixels': changes,
                    'total_pixels': self.total_pixels,
                    'total_bits': capacity['total_bits'],
                    'extracted_bits_count': len(all_extracted_bits)
                }
                
                print(f"\n  РЕЗУЛЬТАТ:")
                print(f"  Сохранено: {output_path}")
                print(f"  Изменено пикселей: {changes} из {self.total_pixels}")
                print(f"  Изменено в %: {changes/self.total_pixels*100:.1f}%")
                print(f"  Извлечено бит всего: {len(all_extracted_bits)}")
                print(f"  Извлечено сообщение: '{extracted_message}'")
                print(f"  Корректно: {extracted_message == message}")
                
            except Exception as e:
                print(f"  ОШИБКА: {e}")
                results[n_bits] = {'error': str(e)}
        
        # Сохраняем отчет
        self.save_report_full(results, message, selected_blocks, output_dir)
        
        return results, selected_blocks, mask
    
    def save_report_full(self, results, original_message, selected_blocks, output_dir):
        """Сохранение подробного отчета о ПОЛНОМ заполнении"""
        report_path = os.path.join(output_dir, "report_full.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 90 + "\n")
            f.write("ОТЧЕТ О ПОЛНОМ ЗАПОЛНЕНИИ 5 БЛОКОВ 32x32\n")
            f.write("=" * 90 + "\n\n")
            
            f.write(f"Исходное сообщение: '{original_message}'\n")
            f.write(f"Длина: {len(original_message)} символов = {len(original_message) * 8} бит\n\n")
            
            f.write("Блоки 32x32:\n")
            for i, (x, y) in enumerate(selected_blocks, 1):
                f.write(f"  Блок {i}: ({x}, {y})\n")
            
            f.write(f"\nВсего пикселей в 5 блоках: {self.total_pixels}\n")
            f.write(f"Всего каналов (RGB): {self.total_pixels * 3}\n\n")
            
            f.write("РЕЗУЛЬТАТЫ ПО УРОВНЯМ (ПОЛНОЕ ЗАПОЛНЕНИЕ):\n")
            f.write("-" * 90 + "\n")
            
            for n_bits in range(1, 9):
                f.write(f"\n[{n_bits} БИТ НА ПИКСЕЛЬ]\n")
                
                if n_bits in results:
                    if 'error' in results[n_bits]:
                        f.write(f"  Статус: ОШИБКА - {results[n_bits]['error']}\n")
                    else:
                        cap = self.capacities[n_bits]
                        f.write(f"  Вместимость: {cap['total_chars']} символов, {cap['total_bits']} бит\n")
                        f.write(f"  Фактически встроено бит: {cap['total_bits']} (ПОЛНОЕ ЗАПОЛНЕНИЕ)\n")
                        f.write(f"  Изменено пикселей: {results[n_bits]['changed_pixels']} из {self.total_pixels}\n")
                        f.write(f"  Изменено в %: {results[n_bits]['changed_pixels']/self.total_pixels*100:.1f}%\n")
                        f.write(f"  Извлечено бит при проверке: {results[n_bits]['extracted_bits_count']}\n")
                        f.write(f"  Извлечено сообщение: '{results[n_bits]['extracted']}'\n")
                        f.write(f"  Корректность: {results[n_bits]['correct']}\n")
                else:
                    f.write(f"  Нет данных\n")
        
        print(f"\nПодробный отчет сохранен: {report_path}")

def demonstrate_full_fill():
    """Демонстрация ПОЛНОГО заполнения всех 5 блоков"""
    
    # Создаем экземпляр класса
    stego = SteganographyMultiLevel(block_size=32, num_blocks=5)
    
    # Тестовое сообщение
    message = "Secret Message"
    
    # Путь к изображению
    input_image = "container.jpg"  # Замените на ваш файл
    
    try:
        # Запускаем обработку с ПОЛНЫМ заполнением
        results, blocks, mask = stego.process_all_levels_full(
            image_path=input_image,
            message=message,
            output_dir="stego_full_fill"
        )
        
        # Итоговая статистика
        print("\n" + "=" * 90)
        print("ИТОГОВАЯ СТАТИСТИКА ПОЛНОГО ЗАПОЛНЕНИЯ")
        print("=" * 90)
        
        for n_bits in range(1, 9):
            if n_bits in results and 'error' not in results[n_bits]:
                cap = stego.capacities[n_bits]
                print(f"\n{n_bits} бит/пиксель:")
                print(f"  Вместимость: {cap['total_chars']} символов")
                print(f"  Встроено бит: {cap['total_bits']} ({cap['total_bits']/8} байт)")
                print(f"  Изменено пикселей: {results[n_bits]['changed_pixels']}")
                print(f"  Сообщение извлечено: '{results[n_bits]['extracted']}'")
                print(f"  Корректно: {results[n_bits]['correct']}")
        
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    demonstrate_full_fill()